---
title: "About Mun Hou"
source: https://blog.munhou.com/about/
date: 2026-11-16
updated: 2026-11-24
---

# About Mun Hou

Hi, I'm Mun Hou. I build and study machine learning systems, and I write here about what I learn along the way.

My recent work is on **LLM agents and reinforcement learning**: training small models to act in tool environments and checking whether a model that predicts its environment better also acts better. Before that I spent a lot of time on **ML for network security** (encrypted traffic classification, intrusion and anomaly detection, malware detection) and on the **data and serving systems** that ML runs on, mostly with PyTorch and PySpark.

I like reimplementing papers to see what holds up. Most of the projects below started that way, and the write-ups say where my results matched the paper and where they didn't.

LLM agents & RL

GRPO, world-model objectives, policy gradient methods, PPO and its successors.

Security ML

Traffic classification, intrusion detection, streaming anomaly detection, out-of-distribution detection, malware.

ML systems

Spark-to-PyTorch pipelines, batch inference, stream-first model serving.

## Featured Project

### [delta\_world](https://github.com/munhouiani/delta_world): world modeling for an LLM agent

An agent that works on Delta Lake tables in a Databricks-style workspace, trained with GRPO and with ECHO, which adds a loss on predicting tool outputs. Policy: NVIDIA Nemotron 3 Nano 4B, trained with NeMo RL on Megatron with vLLM sampling.

-   ECHO-trained models predicted the facts in tool outputs 0.64–0.77 nats per token better than the GRPO model on held-out episodes.
-   ECHO models solved about 0.2 more of the tasks than GRPO (CI +0.12 to +0.29).
-   A better world model did not always mean a better agent, and debugging the training run took most of the effort.

LLM agentsGRPONeMo RLvLLMDelta Lake

[Code](https://github.com/munhouiani/delta_world) [Results](https://blog.munhou.com/World-Modeling-for-LLM-Agent-Predicting-or-Acting-in-the-Environment/) [Lessons learned](https://blog.munhou.com/Lessons-Learned-During-World-Model-Training-for-LLM-Agent/)

[![Dev pass@1 at training steps 0 to 90 for GRPO and two ECHO settings](https://blog.munhou.com/images/echo-model/learning-curve.svg)](https://blog.munhou.com/World-Modeling-for-LLM-Agent-Predicting-or-Acting-in-the-Environment/)

Dev pass@1 during training. With task family E left out, the three curves stay close.

## Projects

### Deep Packet

PyTorch reimplementation of a CNN that classifies encrypted network traffic by application and traffic type, with a PySpark preprocessing pipeline over the ISCX VPN-nonVPN captures.

PyTorchPySparkTraffic classification

[Code](https://github.com/munhouiani/Deep-Packet) [Write-up](https://blog.munhou.com/Pytorch-Implementation-of-Deep-Packet-A-Novel-Approach-For-Encrypted-Traffic-Classification-Using-Deep-Learning/)

### LuNet

PyTorch Lightning implementation of LuNet, which stacks convolution and LSTM blocks to detect network intrusions, trained on the UNSW-NB15 dataset.

PyTorch LightningCNN + LSTMIntrusion detection

[Code](https://github.com/munhouiani/LuNet)

### GEE

Variational autoencoder for network anomaly detection, trained on noisy, unlabeled flow features and scored by reconstruction error, following the GEE paper.

PyTorchVAEAnomaly detection

[Code](https://github.com/munhouiani/GEE) [Write-up](https://blog.munhou.com/Pytorch-Implementation-of-GEE-A-Gradient-based-Explainable-Variational-Autoencoder-for-Network-Anomaly-Detection/)

### MIDAS

Pip-installable Python build of MIDAS, MIDAS-R and MIDAS-F for streaming edge anomaly detection, with an example notebook on a real dataset.

PythonC++Streaming

[Code](https://github.com/munhouiani/MIDAS) [Write-up](https://blog.munhou.com/Network-Anomaly-Detection-with-MIDAS-MIDAS-R-and-MIDAS-F/)

### OOD detection with kNN

Out-of-distribution detection from ResNet-18 embeddings on CIFAR-10, using FAISS nearest-neighbor distances as the score.

PyTorchFAISSOOD

[Code](https://github.com/munhouiani/ood-knn) [Write-up](https://blog.munhou.com/Detecting-Out-of-Distribution-Samples-with-Knn/)

### Rendezvous architecture

Stream-first model serving with production, canary and decoy models behind one front end, message passing over Pulsar, and logs shipped to Elasticsearch.

PulsarElasticsearchDocker

[Code](https://github.com/munhouiani/rendezvous-arch) [Write-up](https://blog.munhou.com/Simple-Implementation-of-Rendevzous-Architecture-for-Machine-Learning-Services/)

### Drug Listings Dataset

A cleaned CSV of drug listings extracted from Gwern's darknet market archives, covering eight marketplaces including Agora, Evolution and Silk Road 2.

DatasetData cleaning

[Data](https://github.com/munhouiani/Drug-Listings-Dataset) [Write-up](https://blog.munhou.com/Drug-Listing-Dataset/)

### Petastorm demo

Feeding Parquet files written by PySpark straight into a PyTorch training loop with Uber's Petastorm.

PySparkPyTorchPetastorm

[Code](https://github.com/munhouiani/petastorm-demo) [Write-up](https://blog.munhou.com/Data-Pipeline-From-Pyspark-to-Pytorch/)

## Writing

### Reinforcement learning, from the ground up

1.  [Importance Sampling](https://blog.munhou.com/Importance-Sampling/)
2.  [Policy Gradient Methods](https://blog.munhou.com/Policy-Gradient-Methods/)
3.  [Off-Policy Gradient Methods](https://blog.munhou.com/Off-Policy-Gradient-Methods/)
4.  [PPO and PPO2](https://blog.munhou.com/Proximal-Policy-Optimization-Algorithms-PPO-and-PPO2/)
5.  [From PPO to GDPO](https://blog.munhou.com/From-PPO-to-GDPO-Understanding-the-Evolution-of-Reinforcement-Learning-Algorithms/)

### Language models

-   [The Raw Output of LLM is a Sequence](https://blog.munhou.com/The-Raw-Output-of-LLM-is-a-Sequence/)
-   [Byte-Pair Encoding (BPE)](https://blog.munhou.com/Byte-Pair-Encoding-BPE/)
-   [Paper study: LLaMA](https://blog.munhou.com/Paper-Study-LLaMA-Open-and-Efficient-Foundation-Language-Models/)
-   [Paper study: LLaMA 2](https://blog.munhou.com/Paper-Study-LLaMA-2-Open-Foundation-and-Fine-Tuned-Chat-Models/)

Everything else is in the [archive](https://blog.munhou.com/archives/) and under [tags](https://blog.munhou.com/tags/).

## Get in Touch

[GitHub](https://github.com/munhouiani) [LinkedIn](https://www.linkedin.com/in/munhouetic/) [Email](mailto:munhou2022+blog@gmail.com)
