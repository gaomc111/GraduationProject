# Research Plan
Target: submit the paper in July 2025
Topic: Temporal GNN link prediction
	Continuous TLP Problem setup: (before Jan. 20)
	Problem Formulation: 

A temporal network can be represented as a sequence of links that come in over time, i.e., \( E = \{(e_1, t_1), (e_2, t_2), \ldots\} \) where \( e_i \) is a link and \( t_i \) is the timestamp showing when \( e_i \) arrives. Each link \( e_i \) corresponds to a dyadic event between two nodes \( \{v_i, u_i\} \). For simplicity, we first assume those links to be undirected and without attributes while later we discuss how to generalize our method to directed attributed networks. The sequence of links encodes network dynamics. Therefore, the capability of a model for representation learning of temporal networks is typically evaluated by how accurately it may predict future links based on the historical links. In this work, we also use link prediction as the metric. Note that we care not only the link prediction between the nodes that have been seen during the training. We also expect the models to predict links between the nodes that have never been seen as the inductive evaluation.

## Transductive Link Prediction Task

The transductive link prediction task allows temporal links between all nodes to be observed up to a time point during the training phase, and uses all the remaining links after that time point for testing. In our implementation, we split the total time range \([0, T]\) into three intervals: \([0, T_{\text{train}})\), \([T_{\text{train}}, T_{\text{val}})\), \([T_{\text{val}}, T]\).

## Inductive Link Prediction Task

The inductive link prediction task predicts links associated with nodes that are not observed in the training set. There are two types of such links:

1. **"old vs. new" links**: Links between an observed node and an unobserved node.
2. **"new vs. new" links**: Links between two unobserved nodes.

Since these two types of links suggest different types of inductiveness, we distinguish them by reporting their performance metrics separately. In practice, we follow two steps to split the data:

1. We use the same setting of the transductive task to first split the links chronologically into training / validation / testing sets.
2. We randomly select 10% nodes, remove any links associated with them from the training set, and remove any links not associated with them in the validation and testing sets.

We randomly sample an equal amount of negative links and consider link prediction as a binary classification problem.

## Baselines

1. **Snapshot-based methods**:
   - DynAERNN (Goyal et al., 2020)
   - VGRNN (Hajiramezanali et al., 2019)
   - EvolveGCN (Pareja et al., 2020)

2. **Stream-based methods**:
   - TGAT (Xu et al., 2020)
   - JODIE (Kumar et al., 2019)
   - DyRep (Trivedi et al., 2019)
   - TGN
   - CAWN
   - TCL
   - GraphMixer
   - DyGFormer

## Datasets

Wikipedia, Reddit, MOOC, LastFM, Enron, Social Evo., UCI, Flights, Can. Parl., US Legis., UN Trade, UN Vote, and Contact.

# Discrete TLP Problem Setup

**High-Quality Temporal Link Prediction for Weighted Dynamic Graphs via Inductive Embedding Aggregation**

A dynamic graph can be represented as a sequence of snapshots \( G = (G_1, \ldots, G_T) \) over time steps \( \{1, 2, \ldots, T\} \). Each snapshot \( G_t \) can be described as \( (V_t, E_t) \), where \( V_t = \{v_1^t, v_2^t, \ldots, v_{N_t}^t\} \) is the node set; \( E_t = \{((v_i^t, v_j^t), w) \mid v_i^t, v_j^t \in V_t, w \in \mathbb{R}^+\} \) is the weighted edge set. We assume that attributes of each node are fixed for all the snapshots, but the topology structure (including the node and edges sets) can change over time. In particular, the fixed node attributes can be information denoting the unique identity of each node in a system (e.g., IP address).

We use an adjacency matrix \( A_t \in \mathbb{R}^{N_t \times N_t} \) to describe the topology of each snapshot \( G_t \), where \( (A_t)_{ij} = (A_t)_{ji} = w > 0 \) if \( ((v_i^t, v_j^t), w) \in E_t \) and \( (A_t)_{ij} = (A_t)_{ji} = 0 \) otherwise. Moreover, we use an attribute matrix \( X_t \in \mathbb{R}^{N_t \times K} \) to describe node attributes of \( G_t \), where the \( i \)-th row \( (X_t)_{i,:} \) is the attribute vector of node \( v_i^t \). For each time step \( t \), we also use an aligning matrix \( B_t \in \mathbb{R}^{N_t \times N_{t+1}} \) to encode the node index correspondence between two successive snapshots \( \{G_t, G_{t+1}\} \). In each snapshot \( G_t \), node indices are renumbered from 1. \( (B_t)_{ij} = 1 \) if node \( v_i^t \) corresponds to \( v_j^{t+1} \) and \( (B_t)_{ij} = 0 \) otherwise (if we allow nodes or edges change with time). In particular, \( B_t \) is an identity matrix when all the snapshots share a common node set.

## Question 1

We assume that all the snapshots share a common node set \( V \), i.e., \( V_1 = \cdots = V_T = V \). Namely, there is no addition and deletion of nodes. We also have \( X_1 = \cdots = X_T = X \), with \( X \) as a common attribute matrix for all the snapshots. Given the topology of previous \( l \) snapshots described by \( A_{\tau-l}^\tau \) and attributes described by \( X \), TLP aims to predict the topology of next time step \( (\tau + 1) \) described by \( \tilde{A}_{\tau+1} \in \mathbb{R}^{|V| \times |V|} \).

## Question 2

We assume that different snapshots can have different node sets. In particular, we only focus on the prediction of weighted edges between nodes observed in previous \( l \) snapshots. For simplicity, we define \( V_{\cup(\tau-l:\tau)} = \cup_{t=\tau-l}^\tau V_t \). Let \( X_{\cup(\tau-l:\tau)} \) be the attribute matrix w.r.t. nodes in \( V_{\cup(\tau-l:\tau)} \). Given the topology and attributes of previous \( l \) snapshots (described by \( A_{\tau-l}^\tau \), \( B_{\tau-l}^\tau \), and \( X_{\tau-l}^\tau \)) as well as the node set and attributes described by \( V_{\cup(\tau-l:\tau)} \) and \( X_{\cup(\tau-l:\tau)} \), TLP aims to predict the topology of next time step w.r.t. \( V_{\cup(\tau-l:\tau)} \).

## Question 3

Not only predict edges between nodes in \( V_{\cup(\tau-l:\tau)} \), but also predict edges (i) between a previously observed node and an unobserved node or (ii) between two unobserved nodes using the prior knowledge of \( \{A_{\tau-l}^\tau, X_{\tau-l}^\tau, X_{\tau+1}\} \). Given the topology and attributes of previous \( l \) snapshots (described by \( A_{\tau-l}^\tau \), \( B_{\tau-l}^\tau \) and \( X_{\tau-l}^\tau \)) as well as the node set and attributes of next time step (described by \( V_{\tau+1} \) and \( X_{\tau+1} \)).

Datasets:
Network dataset: Mesh, Hmob, 
Traffic dateset: DC, T-Drive, 
Social dataset: SEvo, 
Internet: IoT, WIDE,

Baseline: 
CRJMF, DeepEye, TMF, LIST, D2V, DDNE, E-LSTM-D, EGCN, DySAT, STGSN, GCN-GAN, NetGAN

Existing work (Jan. 20-Mar. 7)
Familiar with existing work related to GNN link prediction (MC)
Familiar with existing work related to temporal GNN link prediction (MC)
Run baselines on datasets (MC)
discuss the method (all)

Applying HL-GNN into temporal GNN (Mar. 7-Apr 30)
Summarize the contributions of applying HL-GNN into temporal GNN (all)
Design model that applies HL-GNN into temporal GNN for link prediction and exp (MC)
Think about how to apply the theoretical of HL-GNN to temporal GNN (YN)

Theoretical analysis (May 1-June 1)

Paper writing (June 1-July 1) 

Risk: (1) Extending static GNN to dynamic may be difficult in theoretical analysis. (2) The coding task is heavy and may exceed the expected time. Consider thinking about the theoretical analysis part in the third stage and compressing the time for theoretical analysis and writing.
