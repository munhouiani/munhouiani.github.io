---
title: "Detecting Out-of-Distribution Samples with kNN"
source: https://blog.munhou.com/2022/12/01/Detecting Out-of-Distribution Samples with Knn/
date: 2022-12-01
updated: 2026-11-16
tags: [Deep Learning, PyTorch, Machine Learning]
---

# Detecting Out-of-Distribution Samples with kNN

## What is Out-of-Distribution (OOD) Detection and Why is it Important?

Most machine learning models are trained based on the closed-world assumption, meaning that the test data is assumed to have the same distribution as the training data. When we train a model, we usually form our training and testing data by randomly splitting the data we have into two sets, in which both datasets have the same label distribution. However, in the real-world scenario, this assumption doesn’t necessarily hold. For example, a model trained to classify cats and dogs may receive an image of a dolphin. Even worse, it might highly confident classify such inputs into in-distribution classes.

![OOD Example](https://blog.munhou.com/images/ood-knn/2022-12-01T170823.png)

Detecting if the received model inputs have the same distribution as the training data is called out-of-distribution detection.

OOD detection plays a crucial part in modern-day machine learning services. For example, we can use the OOD detector as part of the machine learning operations to detect unknown inputs. When an unknown is seen, the machine learning service can safely reject the input instead of returning an answer from the model.

![OOD Detector as a prefilter](https://blog.munhou.com/images/ood-knn/2022-12-01T173157.png)

We can also use the result from an OOD detector as a filter for the data labeling task. For example, a machine learning service might receive a lot of inputs every day, and it is not feasible to let human labelers review all data. We can prioritize the process with an OOD detector by reviewing those out-of-distribution samples.

## How is OOD Detection Methods Evaluted?

OOD detection can be seen as a binary classification problem. We often start by training a model with a dataset known as the in-distribution dataset, i.e., CIFAR-10. To evaluate an OOD detector, we will prepare another dataset not in the same domain as the training data, i.e., SHVN, known as the out-of-distribution dataset. We then combine these two datasets to check if the OOD detection knows which is from the in-distribution dataset and vice versa. We usually use label 1 (positive) for in-distribution and 0 (negative) for the out-distribution. Aside from the standard evaluation metrics, we also calculate the FP rate of the OOD detector when its TP rate is at 95%, denoted as FPR@TPR95.

## OOD Detection Methods

From [a comprehensive study by J. Yang et al.](https://arxiv.org/abs/2110.11334), OOD detections methods can be categorized into:

1.  Classification methods
2.  Density-based methods
3.  Distance-based methods
4.  Reconstruction-based methods

## Using kNN as an OOD Detector

The following sections are notes and implementation of a paper by Y. Sun et al.: *Out-of-distribution detection with deep nearest neighbors*.

### Out-of-Distribution Detection for Single Feature

Before we dive into the kNN method, let’s see how out-of-distribution detection works on a single feature. For example, let’s say we have a feature $p$ that is normally distributed with a mean of $0$ and standard deviation of $1$, and $q$ is also drawn i.i.d. from the same distribution.

```python
import numpy as np
from scipy.stats import wasserstein_distance


np.random.seed(7)

# p and q from the same distribution, both has 1000 samples
p = np.random.normal(0, 1, size=1000)
q = np.random.normal(0, 1, size=1000)

# check the wasserstein_distance: 0.04247616759945748
wasserstein_distance(p, q)

# q is drawn from the different distribution
q = np.random.normal(3, 5, size=1000)

# larger wasserstein_distance: 4.170158238494999
wasserstein_distance(p, q)
```

We can use statistical distance measures, such as Wasserstein distance, to check if $p$ and $q$ have the same distribution. If they have the same distribution, the Wasserstein distance will be small. Otherwise, the resulting Wasserstein distance is large.

Traditional statistical distance measures are no longer applied to deep learning models because the inputs are very high-dimensional. Take a $32 \\times 32$ color image as an example, its dimension is $32 \\times 32 \\times 3 = 3072$. Therefore, calculating statistical distance measures for all features is not feasible.

### Core Idea: Embedding

Regarding deep learning models with high dimensional features, current OOD methods focus on the embedding space of a **well-trained** model. Distance-based methods assume that in the embedding space, OOD samples are relatively far away from in-distribution data. Density-based methods believe that in-distribution data follow certain probability distributions. Hence data falls into low-density regions are OOD.

![Embedding space](https://blog.munhou.com/images/ood-knn/2022-12-03T153545.png)

However, these assumptions may not hold. Instead of imposing strong assumptions on the underlying embedding space, Y. Sun et al. use kNN, which is distributional assumption-free. Their algorithm is simple. First, they will gather distances of $k$-nearest neighbor of all in-distribution samples, then use those distances to determine a threshold. Any test data with a kNN distance larger than the threshold are flagged as OOD.

### Train Resnet18 on CIFAR-10

We first train a ResNet18 model with CIFAR-10 data to demonstrate their approaches. This training code is slightly modified from the [official Pytorch Lightning tutorial](https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/cifar10-baseline.html).

```python
class LitResnet(LightningModule):
    def __init__(self, lr=0.05, num_classes=10):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def features(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=num_classes)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                total_steps=self.trainer.estimated_stepping_batches,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
```

```python
model = LitResnet(lr=0.05, num_classes=10)

trainer = Trainer(
    max_epochs=30,
    accelerator="auto",
    devices="auto",
    logger=CSVLogger(save_dir="logs/"),
    callbacks=[
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=10),
    ],
    deterministic=True,
)

trainer.fit(model, train_loader, val_loader)
trainer.test(model, dataloaders=val_loader)
```

This model achieves an accuracy of 94%.

### Feature Extraction

We use the output of the last hidden layer (one layer before the output layer) as the features. Then, we normalize the features with the method described in the paper $\\mathbf{z} = \\phi(\\mathbf{x}) / || \\phi(\\mathbf{x}) ||\_{2}$.

```python
def extract_features(model, dataloader):
    features = []
    for data, label in dataloader:
        # get the last hidden layer's output from model, size: (batch_size, 512, 4, 4)
        x = model.features(data)
        # avg pool, size: (batch_size, 512, 1, 1)
        x = F.adaptive_avg_pool2d(x, 1)
        # squeeze, size: (batch_size, 512)
        x = x.squeeze()
        # detach
        x = x.data.cpu().numpy()

        features.append(x)

    # merge as a large numpy array
    features = np.vstack(features)
    # ensure data type
    features = features.astype(np.float32)
    # normalization
    features = features / (
        np.linalg.norm(features, ord=2, axis=-1, keepdims=True) + 1e-10
    )

    return features
```

```python
train_features = extract_features(model, train_loader)
test_features = extract_features(model, val_loader)
ood_features = extract_features(model, ood_loader)
```

### Find $k$-Nearest Neighbors with FAISS

We now need to find the distances of kNN and determine the threshold. First, we build an index using FAISS with the training features. Then we gather distances of $k$-th neighbors to the testing features. Here we set $k$ to 50. After that, we determine the threshold with 95% of TRP.

```python
# build index with training features
index = faiss.IndexFlatL2(train_features.shape[1])
index.add(train_features)

# find k-th neighbours and distances
k = 50
in_l2_distances, in_k_closest_points = index.search(test_features, k)

# in_l2_distances is sorted increasingly
in_scores = in_l2_distances[:, -1]

# determine threshold with TPR = 0.95
in_scores.sort()
threshold = in_scores[round(0.95 * len(in_scores))]
```

### Evaluate on OOD Data

We follow the same steps as above for the OOD data.

```python
# same as in-distribution data
k = 50
out_l2_distances, out_k_closest_points = index.search(ood_features, k)

# out_l2_distances is sorted increasingly
out_scores = out_l2_distances[:, -1]
```

After getting the distances of $k$-th nearest neighbors of the OOD data, we can easily find the result by checking if the distance is greater than the threshold.

```python
tp = 0
tn = 0
fp = 0
fn = 0

for point in in_scores:
    if point <= threshold:
        tp += 1
    else:
        fn += 1

for point in out_scores:
    if point <= threshold:
        fp += 1
    else:
        tn += 1

tp_rate = tp / (tp + fn)
tn_rate = tn / (tn + fp)
fp_rate = fp / (fp + tn)
fn_rate = fn / (fn + tp)

result = {
    "tp": tp,
    "tn": tn,
    "fp": fp,
    "fn": fn,
    "tpr": tp_rate,
    "fpr": fp_rate,
}
```

The final result has an FPR@TPR95 of 37.45%, higher than the paper’s result of 24.53%. Developing an OOD detector depends on the quality of the model. The paper mentioned that we could improve the result by using contrastive learning. Contrastive learning helps bring data with the same label closer and push data with different labels further, which aligns with the usage of kNN.

Checkout the full notebook here: [https://github.com/munhouiani/ood-knn](https://github.com/munhouiani/ood-knn)

## References

1.  J. Yang, K. Zhou, Y. Li, and Z. Liu, “Generalized out-of-distribution detection: A survey”, *CoRR*, vol. abs/2110.11334, 2021, \[Online\]. Available: [https://arxiv.org/abs/2110.11334](https://arxiv.org/abs/2110.11334)
2.  J. Ren and B. Lakshminarayanan, “Improving Out-of-Distribution Detection in Machine Learning Models”, *Google Research Blog*, Dec. 17, 2019. [https://ai.googleblog.com/2019/12/improving-out-of-distribution-detection.html](https://ai.googleblog.com/2019/12/improving-out-of-distribution-detection.html) (accessed Dec. 02, 2022).
3.  Y. Sun, Y. Ming, X. Zhu, and Y. Li, “Out-of-distribution detection with deep nearest neighbors”, 2022.
4.  PL team, “PyTorch Lightning CIFAR10 ~94% Baseline Tutorial”, *Pytorch Lightning*, Apr. 28, 2022. [https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning\_examples/cifar10-baseline.html](https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/cifar10-baseline.html) (accessed Dec. 02, 2022).
5.  P. Mardziel, “Drift Metrics: How to Select the Right Metric to Analyze Drift”, *Toward Data Science*, Dec. 06, 2021. [https://towardsdatascience.com/drift-metrics-how-to-select-the-right-metric-to-analyze-drift-24da63e497e](https://towardsdatascience.com/drift-metrics-how-to-select-the-right-metric-to-analyze-drift-24da63e497e) (accessed Dec. 03, 2022).
