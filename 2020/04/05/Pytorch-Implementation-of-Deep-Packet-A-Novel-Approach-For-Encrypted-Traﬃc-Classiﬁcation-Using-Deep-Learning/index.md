---
title: "Pytorch Implementation of Deep Packet: A Novel Approach For Encrypted Traﬃc Classiﬁcation Using Deep Learning"
source: https://blog.munhou.com/2020/04/05/Pytorch-Implementation-of-Deep-Packet-A-Novel-Approach-For-Encrypted-Traﬃc-Classiﬁcation-Using-Deep-Learning/
date: 2020-04-05
updated: 2026-11-16
tags: [Deep Learning, PyTorch, PySpark, Traffic Classification]
---

# Pytorch Implementation of Deep Packet: A Novel Approach For Encrypted Traﬃc Classiﬁcation Using Deep Learning

## Why Traffic Classification

The authors explained that network traffic classification attracts many interests in both academia and industrial area is because it is one of the prerequisites for advanced network management task. Network architecture today is designed to be asymmetric, based on the assumption that clients demand download more than upload. However, this assumption doesn’t hold anymore due to the rise of voice over IP (VoIP), P2P, and other symmetric-demand application. Network providers require the knowledge of the application their clients used to allocate adequate resources.

## Related Works

The authors categorised network classification methods into three categories: (1) port-based (2) payload inspection, and (3) statistical machine learning. The summary of the pros and cons of these methods are as below:

1.  Port-based: classifies traffic by the port number in TCP/UDP header
    
    -   Pros: Fast
    -   Cons: Inaccurate, due to port obfuscation, network address translation (NAT), port forwarding, protocol embedding, and random ports assignment
2.  Payload inspection: analyse the payload in the application layer
    
    -   Pros: Accurate
    -   Cons: Pattern-based. Needs to update patterns each time a new protocol is released. Another issue is that this method raises user privacy concern.
3.  Statistical and machine learning: use statistical features of traffic to train a model
    
    -   Pros: Accurate
    -   Cons: Expensive and inefficient as it needs human involved hand-craft features. Slow execution of machine learning model is another concern.

## Dataset

They used the [VPN-nonVPN dataset (ISCXVPN2016)](https://www.unb.ca/cic/datasets/vpn.html). This dataset was captured at the data-link layer. Hence each packet contains an Ethernet header, an IP header, and a TCP/UDP header.

## Pre-processing

During the pre-processing phase, the authors

1.  Remove Ethernet header
2.  Pad traffic with UDP header with zeros to the length of 20 bytes
3.  Mask the IP in the IP header
4.  Remove irrelevant packets such as packets with no payload or DNS packets
5.  Convert the raw packet into a bytes vector
6.  Truncate the vector of size more than 1500, pad zeros for the byte vector less than 1500
7.  Normalise the bytes vector by dividing each element by 255

I used [Scapy](https://scapy.readthedocs.io/en/latest/introduction.html) to modify the packets.

```python
def remove_ether_header(packet):
    if Ether in packet:
        return packet[Ether].payload

    return packet


def mask_ip(packet):
    if IP in packet:
        packet[IP].src = '0.0.0.0'
        packet[IP].dst = '0.0.0.0'

    return packet


def pad_udp(packet):
    if UDP in packet:
        # get layers after udp
        layer_after = packet[UDP].payload.copy()

        # build a padding layer
        pad = Padding()
        pad.load = '\x00' * 12

        layer_before = packet.copy()
        layer_before[UDP].remove_payload()
        packet = layer_before / pad / layer_after

        return packet

    return packet
    
    
def should_omit_packet(packet):
    # SYN, ACK or FIN flags set to 1 and no payload
    if TCP in packet and (packet.flags & 0x13):
        # not payload or contains only padding
        layers = packet[TCP].payload.layers()
        if not layers or (Padding in layers and len(layers) == 1):
            return True

    # DNS segment
    if DNS in packet:
        return True

    return False
```

## Deep Packet

The authors proposed two models. One is CNN, and another is SAE. I only implemented the CNN model, so I introduce only their CNN architecture here.

![](https://blog.munhou.com/images/deep-packet/cnn.png)

The input of their CNN model is a vector of size 1,500. It consists of two consecutive 1-D convolutional layers, followed by a max-pooling layer. Afterwards, the tensor will be flattened and fed into 4 fully connected layers, while the last layer acts as the softmax classifier. The revealed hyperparameters of their convolutional layers, which are

![](https://blog.munhou.com/images/deep-packet/cnn_hyperparameter.png)

However, they didn’t mention the kernel size of the max-pooling layer and the sizes of three dense layers. I set the kernel size to 2, while for the dense layers, I use the setting of the last three layers of their SAE model, which are 200, 100, and 50.

```python
class CNN(LightningModule):
        # two convolution, then one max pool
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=self.hparams.c1_output_dim,
                kernel_size=self.hparams.c1_kernel_size,
                stride=self.hparams.c1_stride
            ),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.hparams.c1_output_dim,
                out_channels=self.hparams.c2_output_dim,
                kernel_size=self.hparams.c2_kernel_size,
                stride=self.hparams.c2_stride
            ),
            nn.ReLU()
        )

        self.max_pool = nn.MaxPool1d(
            kernel_size=2
        )

        # flatten, calculate the output size of max pool
        # use a dummy input to calculate
        dummy_x = torch.rand(1, 1, self.hparams.signal_length, requires_grad=False)
        dummy_x = self.conv1(dummy_x)
        dummy_x = self.conv2(dummy_x)
        dummy_x = self.max_pool(dummy_x)
        max_pool_out = dummy_x.view(1, -1).shape[1]

        # followed by 5 dense layers
        self.fc1 = nn.Sequential(
            nn.Linear(
                in_features=max_pool_out,
                out_features=200
            ),
            nn.Dropout(p=0.05),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(
                in_features=200,
                out_features=100
            ),
            nn.Dropout(p=0.05),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(
                in_features=100,
                out_features=50
            ),
            nn.Dropout(p=0.05),
            nn.ReLU()
        )

        # finally, output layer
        self.out = nn.Linear(
            in_features=50,
            out_features=self.hparams.output_dim
        )
```

## Create Train and Test Data

For each of the application and traffic classification tasks, the dataset is first stratified split into train set and test set with the ratio of 80:20, then each class in the train set are rebalanced by under-sampling. I used all data in the application classification task, but for the traffic classification task, I used only certain apps traffic in each traffic category. This is because I do not know the traffic category of certain apps which are not mentioned in the dataset description page. The applications I used for the traffic classification task are as follow:

| Traffic Category | Applications |
| --- | --- |
| Email | SMPTS, POP3S and IMAPS |
| Chat | ICQ, AIM, Skype, Facebook and Hangouts |
| Streaming | Vimeo and Youtube |
| File Transfer | Skype, FTPS and SFTP |
| VoIP | Facebook, Skype and Hangouts voice calls |
| Torrent | uTorrent and Transmission (Bittorrent) |

## Evaluation Result

![](https://blog.munhou.com/images/deep-packet/cnn_app_classification.png)  
Application classification

![](https://blog.munhou.com/images/deep-packet/cnn_traffic_classification.png)  
Traffic classification

The model performance is closed but not as good as they claimed. This is probably due to the difference between the composition of the train set and the hyperparameter settings.

## Data Model and Code

You can download the train and test set I created at [here](https://drive.google.com/file/d/1_O2LPs3RixaErigJ_WL1Ecq83VXCXptq/view?usp=sharing) and clone the code from [Github](https://github.com/munhouiani/Deep-Packet).

Pre-trained models are available at [here](https://drive.google.com/file/d/1UgSqcN5SG5hqC2imlYu6bB2f9jD1iiu8/view?usp=sharing).
