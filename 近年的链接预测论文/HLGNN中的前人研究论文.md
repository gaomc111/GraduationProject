1. **SEAL**
    > Muhan Zhang and Yixin Chen. 2018. Link prediction based on graph neural networks. Advances in neural information processing systems 31 (2018).
2. **GraIL** 
    > KomalTeru,EtienneDenis,andWillHamilton.2020. Inductiverelationprediction by subgraph reasoning. In International Conference on Machine Learning. PMLR, 9448–9457.
3. **SUREL**  
    > Haoteng Yin, Muhan Zhang, Yanbang Wang, Jianguo Wang, and Pan Li. 2022. Algorithm and system co-design for efficient subgraph-based graph representation learning. arXiv preprint arXiv:2202.13538 (2022).
4. **NBFNet** 
    > Zhaocheng Zhu, Zuobai Zhang, Louis-Pascal Xhonneux, and Jian Tang. 2021. Neural bellman-ford networks: A general graph neural network framework for link prediction. Advances in Neural Information Processing Systems 34 (2021), 29476–29490.

   1. **本文要解决链接预测领域的什么问题**  
      本文要解决的问题是如何在图中有效地进行链接预测。传统的链接预测方法通常基于手工设计的度量，可能不适用于现实世界中的图，而现有的图神经网络方法在可扩展性和可解释性方面存在局限。

   2. **本文使用了什么方法解决这一问题**  
      本文提出了一种基于路径的表示学习框架，称为神经贝尔曼-福特网络（NBFNet）。该方法通过定义节点对的表示为所有路径表示的广义和，同时将每个路径表示定义为路径中边表示的广义积，借助广义贝尔曼-福特算法来高效求解。

   3. **本文认为他有什么独特之处**  
      本文认为NBFNet的独特之处在于它结合了传统路径基方法和现代图神经网络的优点，具备良好的可扩展性、可解释性和高模型容量。此外，NBFNet通过学习操作符来提升路径表示的能力，相比于传统方法，NBFNet在多个数据集上的实验结果显示出显著的性能提升。

5. **Neo-GNN**  
    > Seongjun Yun, Seoyoon Kim, Junhyun Lee, Jaewoo Kang, and Hyunwoo J Kim. 2021. Neo-gnns: Neighborhood overlap-aware graph neural networks for link prediction. Advances in Neural Information Processing Systems 34 (2021), 13683 13694.

    1. 本文要解决链接预测领域的问题是 **GNN（图神经网络）在处理链接预测时表现不佳**，主要因为它们过于依赖平滑的节点特征，而不是图的结构信息，如重叠的邻域、度数和最短路径。
    2. 为了解决这个问题，本文提出了 **Neighborhood Overlap-aware Graph Neural Networks (Neo-GNNs)** 方法。该方法通过从邻接矩阵中学习有用的结构特征，并估计重叠的邻域来进行链接预测。Neo-GNNs 结合了结构信息和输入节点特征，通过一种重叠邻域感知的聚合方案来提高链接预测的准确性。
    3. 本文认为其独特之处在于：
       1. Neo-GNNs **学习结构特征**，而不是依赖手动设计的特征。
       2. 它能够处理 **重叠的多跳邻域**，从而更全面地考虑图的结构信息。
       3. Neo-GNNs 通过 **自适应结合** 来融合结构特征和特征基础的 GNN，提高了模型的表现，超越了传统的 GNN 和启发式方法。

6. **SEG** 
    > Baole Ai, Zhou Qin, Wenting Shen, and Yong Li. 2022. Structure enhanced graph neural networks for link prediction. arXiv preprint arXiv:2201.05293 (2022).

   1. **本文要解决链接预测领域的什么问题：**
      本文旨在解决链接预测中的两个主要问题：如何设计和编码结构特征，以及如何将这些结构特征有效地整合到图神经网络（GNN）中进行链接预测。
   2. **本文使用了什么方法解决这一问题：**
      本文提出了一种名为结构增强图神经网络（SEG）的方法。SEG通过引入路径标记方法（Path Labeling）来捕捉目标节点周围的拓扑信息，并将这些结构特征与常规的GNN模型结合。通过联合训练结构编码器和深度GNN模型，SEG能够融合拓扑结构和节点特征，从而更充分地利用图信息进行链接预测。
   3. **本文认为它有什么独特之处：**
      本文的独特之处在于提出了新的节点位置标记方法（Path Labeling），用于编码图节点的结构特征。此外，SEG框架的设计使得结构特征和节点特征能够在同一嵌入空间中进行融合，从而优化了图信息的使用，提升了链接预测的效果。实验结果显示，SEG在多个OGB链接预测数据集上达到了最先进的结果。

7.  **BUDDY** 
    > Benjamin Paul Chamberlain, Sergey Shirobokov, Emanuele Rossi, Fabrizio Frasca, Thomas Markovich, Nils Hammerla, Michael M Bronstein, and Max Hansmire. 2022. Graph neural networks for link prediction with subgraph sketching. arXiv preprint arXiv:2209.15486 (2022).

   1. **要解决的问题**  
      本文旨在解决图神经网络（GNNs）在链接预测（LP）任务中表现不佳的问题。具体来说，GNNs在表达能力上存在局限，无法有效计数三角形和区分同构节点，这导致其在链接预测中的性能往往低于简单的启发式方法。
   2.  **使用的方法**  
      本文提出了一种新颖的全图GNN模型，称为ELPH（高效链接预测与哈希）。该模型通过传递子图草图作为消息，来近似SGNN（子图GNN）中关键组件，而不需要显式构建子图。此外，本文还开发了一个高度可扩展的模型BUDDY，通过特征预计算来提高效率，避免了在数据集超出GPU内存时的性能损失。
   3.  **独特之处**  
      本文的独特之处在于：
      - ELPH模型在复杂度上与GCN相似，但在表达能力上超过了传统的消息传递GNN（MPNNs），能够有效处理自动同构节点问题。
      - 通过使用子图草图，ELPH能够在不显式构建子图的情况下有效地结合节点特征和图结构，从而提高链接预测的效率和准确性。
      - BUDDY模型使得在数据集较大时也能实现高效的链接预测，解决了传统SGNN在推理时的效率问题。

---

1. **子图GNN模型（如 SEAL、GraIL 和 SUREL）：**  
   - 特点：显式编码节点对周围的子图拓扑信息，采用标记技巧。
   - 局限性：在训练和推理时，需要针对每条边运行一次子图GNN，计算成本较高。

2. **基于源节点的消息传递模型（如 NBFNet 和 REDGNN）：**  
   - 特点：从全局启发式方法中获得灵感，采用基于源节点的消息传递。
   - 局限性：需要为每个源节点训练一个全局GNN，模型训练复杂度较高。

3. **单一全局GNN模型（如 Neo-GNN、SEG 和 BUDDY）：**  
   - **Neo-GNN**：使用两个MLP近似启发式函数。  
   - **SEG**：结合一个GCN层和MLP近似启发式函数。  
   - **BUDDY**：设计了一个基于子图草图消息传递的全新GNN。  
   - 局限性：主要关注局部拓扑信息，难以捕捉全局拓扑特性。

4. **HL-GNN（提出的方法）：**  
   - 特点：能够捕获长达20跳的长距离信息，仅需训练一个全局GNN。
   - 优势：相较其他方法，显著增强了全局拓扑信息的捕获能力，同时提高了训练效率。


1. **启发式方法 (Heuristic-based Approaches)**  
   - **Common-Neighbor Index (CN):**  
     - 特点：通过目标节点的 **共享邻居数量** 计算节点对之间的相似度分数。  
     - 局限：仅利用目标节点的 **两跳邻居** 信息。  
   - **Adamic-Adar (AA):**  
     - 特点：基于节点间的两跳邻居关系进行计算，类似于CN。  
   - **Katz:**  
     - 特点：探索 **高阶邻居关系**，捕捉更多图结构信息。  
   - **Rooted PageRank (PR):**  
     - 特点：基于PageRank算法，衡量节点之间的相似性，通过高阶邻居信息提升性能。  
   - **SimRank (SR):**  
     - 特点：通过衡量 **节点相似性** 进行预测，考虑高阶邻居关系。  
   - **特点：**  
     - 通过简单的图结构特征进行预测，易于理解
     - 泛化能力差
2. **基于表示学习的方法 (Representation Learning-based Approaches)**  
   - **VGAE (Variational Graph Autoencoder):**  
     - 特点：使用GNN对图的结构和节点特征进行编码，生成节点表示；通过 **内积解码器** 预测节点对之间的链接概率。  
   - **SEAL:**  
     - 特点：提取目标节点对的 **子图** 信息，用于预测节点间的链接。  
   - **GNNs（如用于VGAE和SEAL的GNN方法）：**  
     - 特点：学习节点表示，捕获图的 **拓扑结构和节点特征信息**，在链接预测任务中表现优异（state-of-the-art）。  
   - **特点：**  
     - 依赖于深度学习技术，如GNN，具有更强的预测能力
     - 缺乏可解释性，难以在需要解释链接预测依据的场景中应用。
