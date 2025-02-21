# 基线方法测试

## Datasets & Baseline
**Datasets**：
Network dataset: [Mesh](https://crawdad.org/ucsb/meshnet/20070201/), [Hmob](https://crawdad.org/ncsu/mobilitymodels/20090723/), 
Traffic dateset: [DC](https://github.com/shouxi/numfabric), [T-Drive](https://www.microsoft.com/en-us/research/publication/t-drive-drivingdirections-based-on-taxi-trajectories/ ), 
Social dataset: [SEvo](http://realitycommons.media.mit.edu/socialevolution.html ), 
Internet: [IoT](https://iotanalytics.unsw.edu.au/iottraces.html), [WIDE](https://mawi.wide.ad.jp/mawi/),


**Baseline**：
- **OTI**: CRJMF, DeepEye, TMF, LIST
- **OTOG**: D2V, DDNE, E-LSTM-D, EGCN, DySAT, STGSN, GCN-GAN, NetGAN

![alt text](image.png)
其中：
- “L1”、“L2”和“L3”代表第III部分定义的TLP的三个级别
- “S”表示该方法只能处理级别2的特殊情况（即使用大型邻接矩阵表示可能包含孤立节点的快照）
- “Param”表示待优化模型参数的空间复杂度
- “Res”表示表示预测结果 $\tilde{A}_{\tau+1}$ 的空间复杂度
- $N_C$ 和 $N_t$ 与表I中的定义相同； $N_U = |]\mathcal{V}_{\cup(\tau−l:\tau)}|$ 是前 $l$ 个快照的累积节点数；d是潜在嵌入的维度。通常，我们有 $d < N_t \le N_U \le N_C$

## 测试结果