### Research Plan
#### Target: submit the paper in July 2025
#### Topic: Temporal GNN link prediction

- **Problem setup: (before Jan. 20)**
  - Problem Formulation: 
    A temporal network can be represented as a sequence of links that come in over time, i.e. \( E={(e_1,t_1),(e_2,t_2),...} \) where \( e_i \)  is a link and \( t_i \) is the timestamp showing when \( e_i \) arrives. Each link \( e_i \) corresponds to a dyadic event between two nodes \( {v_i  ,u_i} \). For simplicity, we first assume those links to be undirected and without attributes while later we discuss how to generalized our method to directed attributed networks. The sequence of links encodes network dynamics. Therefore, the capability of a model for representation learning of temporal networks is typically evaluated by how accurately it may predict future links based on the historical links. In this work, we also use link prediction as the metric. Note that we care not only the link prediction between the nodes that have been seen during the training. We also expect the models to predict links between the nodes that has never been seen as the inductive evaluation.
  - Transductive link prediction task allows temporal links between all nodes to be observed up to a time point during the training phase, and uses all the remaining links after that time point for testing. In our implementation, we split the total time range \( [0, T] \) into three intervals: \( [0,T_{train}), [T_{train},T_{val}), [T_{val},T] \). links occurring within each interval are dedicated to training, validation, and testing set, respectively.
  - Inductive link prediction task predicts links associated with nodes that are not observed in the training set. There are two types of such links: 1) "old vs. new" links, which are links between an observed node and an unobserved node; 2) "new vs. new" links, which are links between two unobserved nodes. Since these two types of links suggest different types of inductiveness, we distinguish them by reporting their performance metrics separately. In practice, we follow two steps to split the data: 1) we use the same setting of the transductive task to first split the links chronologically into training / validation / testing sets; 2) we randomly select 10% nodes, remove any links associated with them from the training set, and remove any links not associated with them in the validation and testing sets.
  - randomly sample an equal amount of negative links and consider link prediction as a binary classification problem.
- **Baselines**:
  - (1) Snapshot-based methods, including DynAERNN (Goyal et al., 2020), VGRNN (Hajiramezanali et al., 2019) and EvolveGCN (Pareja et al., 2020); 
  - (2) Stream-based methods, including TGAT (Xu et al., 2020), JODIE (Kumar et al., 2019) and DyRep (Trivedi et al., 2019), TGN, CAWN, TCL, GraphMixer, DyGFormer
- **Datasets**:
  - Wikipedia, Reddit, MOOC, LastFM, Enron, Social Evo., UCI, Flights, Can. Parl., US Legis., UN Trade, UN Vote, and Contact

- **Existing work (Jan. 20-Mar. 7)**
  - Familiar with existing work related to GNN link prediction (MC)
  - Familiar with existing work related to temporal GNN link prediction (MC)
  - Run baselines on datasets (MC)
  - discuss the method (all)

- **Applying HL-GNN into temporal GNN (Mar. 7-Apr 30)**
  - Summarize the contributions of applying HL-GNN into temporal GNN (all)
  - Design model that applies HL-GNN into temporal GNN for link prediction and exp (MC)
  - Think about how to apply the theoretical of HL-GNN to temporal GNN (YN)

- **Theoretical analysis (May 1-June 1)**

- **Paper writing (June 1-July 1)**

Risk: (1) Extending static GNN to dynamic may be difficult in theoretical analysis. (2) The coding task is heavy and may exceed the expected time. Consider thinking about the theoretical analysis part in the third stage and compressing the time for theoretical analysis and writing.
