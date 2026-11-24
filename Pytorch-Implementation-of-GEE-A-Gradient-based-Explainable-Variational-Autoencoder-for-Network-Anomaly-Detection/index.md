---
title: "Pytorch Implementation of GEE: A Gradient-based Explainable Variational Autoencoder for Network Anomaly Detection"
source: https://blog.munhou.com/Pytorch-Implementation-of-GEE-A-Gradient-based-Explainable-Variational-Autoencoder-for-Network-Anomaly-Detection/
date: 2020-07-12
updated: 2026-11-24
tags: [Deep Learning, PyTorch, Machine Learning, PySpark]
---

# Pytorch Implementation of GEE: A Gradient-based Explainable Variational Autoencoder for Network Anomaly Detection

## Motivation

Recently I have been studying on varies methods of anomaly detection, ranging from the traditional methods, such as Isolation Forest to the latest deep-neural-network-based methods. All these methods have their beauty and shortcoming. The reason why I selected and implemented this paper, [GEE: A Gradient-based Explainable Variational Autoencoder for Network Anomaly Detection](https://arxiv.org/abs/1903.06661), is because it used an autoencoder trained with incomplete and noisy data for an anomaly detection task.

![](https://blog.munhou.com/images/gee/autoencoder.png)  
Image by Arden Dertat via [Toward Data Science](https://towardsdatascience.com/applied-deep-learning-part-3-autoencoders-1c083af4d798)

The news which an autoencoder can be used for anomaly detection has been circulating on the internet for a while. Autoencoder was originally designed to learn the latent representation of a bunch of data. It consists of two parts, the encoder and decoder. Autoencoder will first encode the input data to a lower dimension of representation, it then decodes the representation back to its original form. Therefore, ones may suggest that if the difference between the reconstruction of a sample and its original form is big, then that sample could be an anomaly. Many internet articles suggested this idea, but only a few of them realised it. Therefore, I would like to implement this idea myself. By the way, [Joseph Rocca wrote an exceptionally well introductory of autoencoders](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73). If you are not familiar with the idea of an autoencoder, you are welcomed to read it.

![](https://blog.munhou.com/images/gee/richard_feynman.png)  
Image by Riza Fahmi via [Twitter](https://twitter.com/rizafahmi22/status/1000008342859894784).

The other reason I implemented this is that the data it used wasn’t audio-like or image-like data. We’ve seen so many successful stories of deep learning applying on image or audio data via end-to-end learning, i.e., without human-involved feature engineering. I wonder if a deep neural network can be applied on traditional table-like data as well. This paper used NetFlow data, which already discard much raw information of a packet. Furthermore, it grouped these NetFlow data with a specified time window and source IP then performed feature extraction on the grouped data. It would be interesting to see how a deep neural network performs on this kind of feature engineered data.

## Feature Extraction

As mentioned in the paper, the authors first group the NetFlow records into 3-minute sliding windows based on the source IP address and compute the aggregated features. Here is the list of features extracted:

-   mean and standard deviation of ﬂow durations, number of packets, number of bytes, packet rate, and byte rate;
-   entropy of protocol type, destination IP addresses, source ports, destination ports, and TCP ﬂags; and
-   proportion of ports used for common applications.  
    For the ports of common applications, as they didn’t mention which exact ports they chose, I selected these ports:
    -   DHCP
    -   DNS
    -   FTP
    -   HTTP
    -   HTTPS
    -   IMAP
    -   IPSec
    -   LDAP
    -   NetBios
    -   NNTP
    -   NTP
    -   POP3
    -   RDP
    -   RPC
    -   SMTP
    -   SNMP
    -   SSH
    -   Telnet
    -   TFTP
    -   Other

The final number of features will be 69 instead of the original 53 (5 mean-related + 5 std-related + 5 entropy-related + 22 source-port-proportion + 22 dest-port-proportion).

I used PySpark to accelerate the extraction process.

```python
def extract_features(self) -> pyspark.sql.dataframe:
    df = (
        self.df
            # group by src_ip and time_window as in paper
            .groupby('time_window', 'src_ip')
            # start extracting feature
            .agg(
            self.extract_num_flow_udf(lit(1)).alias('num_flow'),
            self.mean_udf('duration').alias('mean_duration'),
            self.mean_udf('packet').alias('mean_packet'),
            self.mean_udf('num_of_bytes').alias('mean_num_of_bytes'),
            self.mean_udf('packet_rate').alias('mean_packet_rate'),
            self.mean_udf('byte_rate').alias('mean_byte_rate'),
            self.std_udf('duration').alias('std_duration'),
            self.std_udf('packet').alias('std_packet'),
            self.std_udf('num_of_bytes').alias('std_num_of_bytes'),
            self.std_udf('packet_rate').alias('std_packet_rate'),
            self.std_udf('byte_rate').alias('std_byte_rate'),
            self.entropy_udf('protocol').alias('entropy_protocol'),
            self.entropy_udf('dst_ip').alias('entropy_dst_ip'),
            self.entropy_udf('src_port').alias('entropy_src_port'),
            self.entropy_udf('dst_port').alias('entropy_dst_port'),
            self.entropy_udf('flags').alias('entropy_flags'),
            self.port_proportion_udf('src_port').alias('proportion_src_port'),
            self.port_proportion_udf('dst_port').alias('proportion_dst_port'),
            self.build_label_udf('label').alias('label'),
        )
            # filter out num_flow < 10
            .filter((col('num_flow') >= 10))
            # sort by time window and source ip
            .orderBy('time_window', 'src_ip')
            # drop num_flow
            .drop('num_flow')
            # fill na
            .na.fill(0.0)
    )

    return df
```

## Build Input for Model

This step is simple, I scaled the extracted with a min-max scaler, made all the values ranged from 0 to 1. Besides that, I also converted the data schema to Petastorm compatible schema. Petastorm enables our learning process directly applied to the data generated by big data processing platform, such as PySpark.

```python
def transform(self, remove_malicious=True, remove_null_label=True) -> pyspark.sql.DataFrame:
    df = (
        self.df
            # compose feature
            .withColumn('features', self.feature_compose_udf(*self.feature_column))
    )

    if remove_null_label:
        df = df.filter(col('label').isNotNull())

    if remove_malicious:
        df = df.filter(col('label') == 'background')

    # select only time_window, src_ip, feature and label columns
    df = df.select(
        'time_window', 'src_ip', 'features', 'label',
    )

    return df
```

## Model

This paper used a VAE model (Variational autoencoder).

```python
class VAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def training_step(self, batch, batch_idx):
        x = batch['feature']
        recon_x, mu, logvar = self(x)
        loss = self.loss_function(recon_x, x, mu, logvar)

        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.01)
```

## Evaluation

Here comes the interesting part. We say that a sample is anomalous if its reconstruction error is big. Here lie two questions: how big is big, and how to compute the reconstruction error.

To compute the reconstruction error, instinctively, I would choose the same loss function from the model, which is binary cross-entropy with or without the regularisation term, the KL-divergence. However, neither of these are performing well. Before I started to give up, I used the L2 distance (MSE) as the reconstruction error. Boom! Surprisingly, it did a lot better the two previously mentioned loss.

![](https://blog.munhou.com/images/gee/roc_background_spam.png)

Now we found the best way to compute reconstruction error. How do we decide the threshold? Let me plot the distribution of errors from normal and malicious samples.

![](https://blog.munhou.com/images/gee/kde_background_spam.png)

From the above image, we can see that the normal samples and malicious samples have two almost different error distribution, where the errors of the normal samples mostly concentrate at around 0.00, while the malicious ones concentrate at around 0.02. Based on the image, we can then set a hard threshold at 0.015. [Balabit unsupervised wrote a detailed article on calibrating a threshold that may worth a read.](https://medium.com/balabit-unsupervised/calibrating-anomaly-scores-5e60b7e47553)

![](https://blog.munhou.com/images/gee/gradient_spam.png)

The authors also introduce a method to explain the VAE by the gradients. The explanation is somehow not very convincible, so I didn’t do a further experiment.

## Code and Repo

Feel free to look on the codes. [https://github.com/munhouiani/GEE](https://github.com/munhouiani/GEE)
