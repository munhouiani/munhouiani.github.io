---
title: "Simple Implementation of Rendezvous Architecture for Machine Learning Services"
source: https://blog.munhou.com/Simple-Implementation-of-Rendevzous-Architecture-for-Machine-Learning-Services/
date: 2022-02-08
updated: 2026-11-23
tags: [Machine Learning]
---

# Simple Implementation of Rendezvous Architecture for Machine Learning Services

## Stream-First Rendezvous Architecture

The rendezvous architecture was introduced by Ted Dunning in his book [Machine Learning Logistics](https://www.oreilly.com/library/view/machine-learning-logistics/9781491997628/ch03.html). This architecture is aimed to solve several real-world machine learning challenges:

-   To meet large-scale changing data and changing goals.
-   Isolation of models for evaluation in specifically customized, controlled environments.
-   Managing multiple models in production at any given time.
-   Have new models being readied to replace production models as situations change smoothly and without interruptions to service.

![](https://blog.munhou.com/images/rendezvous-arch/stream_first_arch.png)  
The heart of rendezvous architecture is to treat every input data as stream, i.e.:

-   Put all requests into a stream; consumers (models) process input data when needed.
-   Outputs of models are put into another stream.
-   Rendezvous server works by maintaining a mailbox for each request it sees in the input stream. As each model reports results into the scores stream, the rendezvous server reads these results and inserts them into the corresponding mailbox.
-   Based on the amount of time that has passed, the priority of each model and possibly even a random number, the rendezvous server eventually chooses a result for each pending mailbox and packages that result to be sent as a response to the return address in the original request.

If you are interested the details of rendezvous architecture for machine learning, I highly recommend reading the book [Machine Learning Logistics](https://www.oreilly.com/library/view/machine-learning-logistics/9781491997628/) and an article [Rendezvous Architecture for Data Science in Production](https://towardsdatascience.com/rendezvous-architecture-for-data-science-in-production-79c4d48f12b) written by Jan Teichmann.

## Implementation

### Architecture Diagram

![](https://blog.munhou.com/images/rendezvous-arch/flow.png)

### `predict` The Front End

```python
def predict(flower: Flower):
    _id = str(uuid.uuid4())
    message = {
        "timestamp": float(datetime.datetime.now(datetime.timezone.utc).timestamp()),
        "messageId": _id,
        "modelInput": {
            "sepalLength": flower.sepalLength,
            "sepalWidth": flower.sepalWidth,
            "petalLength": flower.petalLength,
            "petalWidth": flower.petalWidth,
        },
        "scoreTopic": ret_topic,
    }

    publish_data(message)
    result = get_model_result(_id)

    message["modelResult"] = result
    message.pop("scoreTopic", None)

    return message
```

Once `predict` receives a request, it publishes the request body to a topic, i.e., the `iris`. Then it waits for the result to be published to another topic, `score`.

```python
def get_model_result(identifier):
    now = datetime.datetime.utcnow()

    result = {}

    while (datetime.datetime.utcnow() - now) < datetime.timedelta(seconds=timeout):
        try:
            msg = consumer.receive(2)
            data = json.loads(msg.data())
            logger.debug(f"Receive message: {data}")
            consumer.acknowledge(msg)
            if data["messageId"] != identifier:
                continue
            if not result or (
                data["provenance"]["model"]["modelVersion"]
                > result["provenance"]["model"]["modelVersion"]
            ):
                result = data
        except _pulsar.Timeout:
            continue
    model_result = result.get("result", {})
    logger.debug(f"Model result: {model_result}")
    return model_result
```

`get_model_result` gets results from the `score` topic and will only keep the result from the latest model by comparing the `modelVersion`. We also impose a time constraint `timeout` here to ensure our service responds in time.

### Prediction Models

```python
while True:
    msg = consumer.receive()
    data = json.loads(msg.data())
    logger.debug(f"Receive message: {data}")

    # get ret addr
    ret_topic = data["scoreTopic"]
    producer = get_producer(client, ret_topic)
    logger.debug(f"Create a producer for topic {ret_topic}")

    result = get_model_result(model, data)
    request_timestamp = datetime.datetime.fromtimestamp(
        float(data["timestamp"]), datetime.timezone.utc
    )
    now_timestamp = datetime.datetime.now(datetime.timezone.utc)
    time_after_request = (now_timestamp - request_timestamp).total_seconds() * 1000

    data["result"] = {
        "modelName": model_name,
        "result": result,
    }

    if "provenance" not in data:
        data["provenance"] = {}

    data["provenance"]["model"] = {
        "modelName": model_name,
        "elapsedTimeAfterRequest": time_after_request,
        "modelVersion": model_version,
    }

    producer.send(json.dumps(data).encode("utf-8"))
    producer.flush()
    producer.close()
    logger.debug(f"Sent data with producer: {data}")

    consumer.acknowledge(msg)
```

We have two models, the `production model` is the latest, and the `canary model` is the baseline model. We’re trying to recreate the real-world scenario where we have a production model that is actively trained with the latest data, providing better performance. On the other hand, the canary model is usually the first model, intended to be used as a scoring baseline to compare with the production model. By comparing these two models, we can usually detect distribution shifts.

This model predict function will process every data from the `iris` topic and put model result and relevant debug message into the `provenance` data field.

### Decoy Model

```python
while True:
    msg = consumer.receive()
    data = msg.data()
    logger.debug(f"Receive message: {data}")
    producer.send(data)
    logger.debug(f"Sent data with producer: {data}")
    consumer.acknowledge(msg)
```

The decoy model accepts data like any other model but does not emit any result. Instead, it just archives the inputs that it sees.

### Log Connector

```python
while True:
    time_diff = datetime.datetime.utcnow() - now
    is_timeout = time_diff > timeout
    if len(logs) >= threshold or is_timeout:
        if is_timeout:
            now = datetime.datetime.utcnow()
        if not logs:
            continue

        filename = f"log-{uuid.uuid4()}.gz"
        filepath = os.path.join(output_path, filename)
        logger.debug(
            f"output log to {filepath} timeout: {is_timeout}, length: {len(logs)}"
        )
        with gzip.open(filepath, "wb") as f_out:
            for log in logs:
                log = json.loads(log)
                f_out.write(f"{json.dumps(log)}\n".encode("utf-8"))
            logs.clear()
    try:
        msg = consumer.receive(2)
        data = msg.data()
        logger.debug(f"Received data {data}")
        logs.append(data)
        consumer.acknowledge(msg)
    except:
        continue
```

The `log_collector` archives every data we received for every `timeout` second.

### ES Connector

```python
while True:
    msg = consumer.receive()
    data = json.loads(msg.data())
    # fix timestamp
    data["timestamp"] = datetime.datetime.fromtimestamp(
        int(data["timestamp"]), tz=datetime.timezone.utc
    )
    logger.debug(f"Received data {data}")
    res = es.index(index=elastic_index, body=data)
    consumer.acknowledge(msg)
```

We also put every result to the elastic search for analytic purposes.

## How to Run

Clone my repo here at [https://github.com/munhouiani/rendezvous-arch](https://github.com/munhouiani/rendezvous-arch)

1.  Install docker and docker compose.
    
2.  Create conda environment
    
    ```bash
    conda env create -f env_mac.yaml
    conda activate rendezvous_arch
    ```
    
3.  Train `canary` and `production` model
    
    ```bash
    make train-canary
    ```
    
    ```bash
    make train-model
    ```
    
4.  Deploy services
    
    ```bash
    make up
    ```
    
    Wait for several minutes until all services are ready.
    
5.  Make an HTTP-GET request to `http://localhost:8000/ping`, should get
    
    ```json
    {"ping": "pong"}
    ```
    
6.  Make an HTTP-POST request to `localhost:8000/predict` with body:
    
    ```json
    {
       "sepalLength": 9,
       "sepalWidth": 10,
       "petalLength": 11,
       "petalWidth": 12
    }
    ```
    
    should get
    
    ```json
    {
        "timestamp": 1644370787.021771,
        "messageId": "cec42337-49ad-4f19-a478-69a1ef480e8a",
        "modelInput": {
            "sepalLength": 9.0,
            "sepalWidth": 10.0,
            "petalLength": 11.0,
            "petalWidth": 12.0
        },
        "modelResult": {
            "modelName": "dt",
            "result": 2
        }
    }
    ```
    
7.  Create indices for `decoy-log` and `score-log` at Kibana `http://localhost:5601`.
    

## References

1.  [Streaming Architecture including Rendezvous for Machine Learning](https://www.slideshare.net/tdunning/streaming-architecture-including-rendezvous-for-machine-learning)
2.  [Machine Learning Logistics](https://www.oreilly.com/library/view/machine-learning-logistics/9781491997628/)
3.  [Rendezvous Architecture for Data Science in Production](https://towardsdatascience.com/rendezvous-architecture-for-data-science-in-production-79c4d48f12b)
