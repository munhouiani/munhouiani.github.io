---
title: "ML Prediction with XGBoost and PySpark"
source: https://blog.munhou.com/ML-Prediction-with-XGBoost-and-PySpark/
date: 2020-02-29
updated: 2026-11-25
tags: [Machine Learning, PySpark]
---

# ML Prediction with XGBoost and PySpark

Once a XGBoost model is trained, we would like to use PySpark for batch predictions.

The method we use here is through Pandas UDF.

## Why `pandas_udf` Instead of `udf`

For a `udf` function, PySpark evaluates it one record at a time, which is the slowest possible way to execute the prediction. While for a `pandas_udf` function, it takes a bunch of pandas `Series` and returns a `Series`, which is vectorised.

## Create a `pandas_udf` Prediction Function

```python
@pandas_udf(FloatType())
def get_prediction_result(features):
    # convert to 2D-numpy array
    numpy_array = np.array(
        features
        .values
        .tolist()
    )
    # convert to xgboost dmatrix
    dmatrix = xgb.DMatrix(numpy_array)
    
    # predict
    y = model.predict(dmatrix)
    
    # return with pandas series
    series = pd.Series(y)
    return series
```

Let’s say we have a dataframe `df` with following records:

| features |
| --- |
| \[149.0, 84.0. 38.5, 79.7\] |
| \[2.5, 34.0. 35.5, 97.6\] |
| \[0.0, 65.3. 58.7, 45.7\] |

`features` is the feature column with `ArrayType(FloatType())` and each row of `features` contains 4 feature values.

When we call `get_prediction_result` function, it receives a `Series` object. In this case, it would be

```plaintext
0    [149.0, 84.0, 38.5, 79.7]
1    [2.5, 34.0, 35.5, 97.6]
2    [0.0, 65.3, 58.7, 45.7]
dtype: object
```

We would need to convert it to a `DMatrix` object, which is the input for a XGBoost model. We first call its `values` method, which yields:

```plaintext
array([list([149.0, 84.0, 38.5, 79.7]), list([2.5, 34.0, 35.5, 97.6]),
       list([0.0, 65.3, 58.7, 45.7])], dtype=object)
```

Then we call `tolist` to convert it into a list of list, yields:

```plaintext
[[149.0, 84.0, 38.5, 79.7], [2.5, 34.0, 35.5, 97.6], [0.0, 65.3, 58.7, 45.7]]
```

After that, we convert it into a 2D numpy array:

```plaintext
array([[149. ,  84. ,  38.5,  79.7],
       [  2.5,  34. ,  35.5,  97.6],
       [  0. ,  65.3,  58.7,  45.7]])
```

Finally, we initialise a `DMatrix` object with the 2D numpy array we just created. We then run the prediction. As a `pandas_udf` requires a `Series` object to return, we convert the prediction scores into a `Series` at the line 16.

## Using a `pandas_udf` Function

Using a `pandas_udf` function is just like how we use a normal `udf` function:

```python
df.withColumn('prediction', get_prediction_result('features'))
```

That’s how we integrate the XGBoost batch prediction into our PySpark pipeline.
