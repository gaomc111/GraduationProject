# 基线方法测试

## Datasets & Baseline
**Datasets**：
Network dataset: [Mesh](https://crawdad.org/ucsb/meshnet/20070201/), [Hmob](https://crawdad.org/ncsu/mobilitymodels/20090723/), 
Traffic dateset: [DC](https://github.com/shouxi/numfabric), [T-Drive](https://www.microsoft.com/en-us/research/publication/t-drive-drivingdirections-based-on-taxi-trajectories/ ), 
Social dataset: [SEvo](http://realitycommons.media.mit.edu/socialevolution.html ), 
Internet: [IoT](https://iotanalytics.unsw.edu.au/iottraces.html), [WIDE](https://mawi.wide.ad.jp/mawi/),


**Baseline**：
- **OTI**: CRJMF, DeepEye, TMF, LIST
- **OTOG**: D2V, DDNE, E-LSTM-D, EvolveGCN(EGCN), DySAT, STGSN, GCN-GAN, NetGAN

![alt text](image.png)
其中：
- “L1”、“L2”和“L3”代表第III部分定义的TLP的三个级别
- “S”表示该方法只能处理级别2的特殊情况（即使用大型邻接矩阵表示可能包含孤立节点的快照）
- “Param”表示待优化模型参数的空间复杂度
- “Res”表示表示预测结果 $\tilde{A}_{\tau+1}$ 的空间复杂度
- $N_C$ 和 $N_t$ 与表I中的定义相同； $N_U = |]\mathcal{V}_{\cup(\tau−l:\tau)}|$ 是前 $l$ 个快照的累积节点数；d是潜在嵌入的维度。通常，我们有 $d < N_t \le N_U \le N_C$

**论文指标**：
![alt text](image-1.png)

## 基线方法实现情况
- **CRJMF**：
  - Temporal link prediction by integrating content and structure information
  - 无
- **DeepEye**：
  - DEEPEYE: Link Prediction in Dynamic Networks Based on Non-negative Matrix Factorization
  - 无
- **TMF**:
  - Temporally Factorized Network Modeling for Evolutionary Network Analysis
  - 无
- **LIST**：
  - Link prediction with spatial and temporal consistency in dynamic Networks
  - 无
- **D2V**:
  - **部分实现**，DynamicGEM
- **DDNE**：
  - Deep Dynamic Network Embedding for Link Prediction
  - 无
- **E-LSTM-D**：
  - E-LSTM-D: A Deep Learning Framework for Dynamic Network Link Prediction
  - **有实现**，tensorflow
- **EvolveGCN**:
  - **有实现**
- **DySAT**:
  - Dysat: Deep neural representation learning on dynamic graphs via self-attention networks
  - **上古实现**，python2.7 + tensorflow
- **STGSN**：
  - STGSN — A Spatial–Temporal Graph Neural Network framework for time-evolving social networks
  - 无
- **GCN-GAN**：
  - GCN-GAN: A Non-linear Temporal Link Prediction Model for Weighted Dynamic Networks
  - **民间实现**，报一堆错
- **NetGAN**：
  - An Advanced Deep Generative Framework for Temporal Link Prediction in Dynamic Networks
  - 无

## 测试结果
- **E-LSTM-D**：
  - 1
- **EvolveGCN**:
  - 能运行
  - [EvolveGCN](EvolveGCN.txt)
  - [EvolveGCN](EvolveGCN2.txt)
- **DySAT**:
  - 新conda环境`DySAT`
  - 能运行
  - [DySAT](DySAT.txt)