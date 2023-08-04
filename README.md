# MLGAL-Multi-level-Label-Graph-Adaptive-Learning-for-Node-Clustering-in-the-Attributed-Graph
The implementation of the MLGAL-Multi-level-Label-Graph-Adaptive-Learning-for-Node-Clustering-in-the-Attributed-Graph

## Introduction
Node clustering aims to divide nodes into disjoint groups. Recently, aconsiderable amount of research leverages Graph Neural Networks (GNNs) to learn compact node embeddings, which are then used as input of the traditional clustering methods to get better clustering results. While in most of these methods the node representation learning and the clustering are performed separately, a few works have further coupled them in a self-supervised learning manner. The coupling, however, should be carefully designed to avoid potential noises in the pseudo labels generated automatically during the training process.
To address the above problems, in this research, we propose Multi-level Label Graph Adaptive Learning (MLGAL), a novel unsupervised learning algorithm for the node clustering problem. We first design a graph filter to smooth the node features. Then, we iteratively choose the similar and the dissimilar node pairs to perform the adaptive learning with the multi-level label, i.e., the node-level label and the cluster-level label generated automatically by our model

## How to Use
### Unzip the Data.zip
```
unzip dataset.zip
```

### Run the Dataset
```
python main.py --dataset cora --pos_st 0.002 --pos_ed 0.001 --neg_st 0.9 --neg_st 0.5
```
