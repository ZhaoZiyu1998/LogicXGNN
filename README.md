# 🧩 LOGICXGNN: GROUNDED LOGICAL RULES FOR EXPLAINING GRAPH NEURAL NETWORKS

This is the official repository of **LogicXGNN**.

---

## 📌 Features

- Multiple GNN backbones:
  - **GCN**, **GIN**, **GAT**, **GraphSAGE**, and etc.
- Wide range of datasets:
  - **Molecule datasets**: BBBP, Mutagenicity, NCI1  
  - **Synthetic**: BAMultiShapes  
  - **Social**: IMDB-BINARY, Reddit, Twitch, GitHub stargazers
- Explanation modules:
  - Derive **predicates** from GNN embeddings and the underlying message-passing topology.
  - Extract **logical rules** over these predicates to approximate and interpret GNN decisions.
  - Efficiently **ground predicates** into representative subgraphs, supported by general grounding rules.

- Training utilities:
  - Class-balanced weights
  - Checkpoint saving/loading
  - Early stopping thresholds per dataset

---

## 📂 Repository Structure

```
repo/
├── load_data.py        # Dataset loaders and preprocessing
├── gnn.py             # GCN, GIN, GAT, GraphSAGE implementations
├── main.py            # Main training + explanation pipeline
├── utils.py           # Training, testing, model checkpointing
├── explain_gnn.py     # Activation extraction + decision tree explainer
├── grounding.py       # Grounding predicates
├── models/            # Saved checkpoints (created at runtime)
├── plot/              # Generated plots (created at runtime)
└── readme/            # This file
```

---

## 🚀 Installation

Clone the repo and set up dependencies:

```bash
wget https://anonymous.4open.science/r/LogicXGNN-ICRL2026/
cd LogicXGNN-ICRL2026

conda create -n logicgnn python=3.10
conda activate logicgnn
pip install torch torch-geometric scikit-learn matplotlib networkx
```

---

## 🛠 Usage

### Training a Model

```bash
python main.py --dataset BBBP --arch GCN --seed 42
```

### Arguments

- `--dataset`: one of `BBBP`, `Mutagenicity`, `IMDB-BINARY`, `NCI1`, `BAMultiShapes`, `reddit_threads`, `twitch_egos`, `github_stargazers`
- `--arch`: one of `GCN`, `GIN`, `GAT`, `GraphSAGE`
- `--seed`: random seed (default: 0)
- `--load`: load a pretrained model
- `--max_depth`: maximum depth for decision tree explanations

### Example

Train a GIN on Mutagenicity:

```bash
python main.py --dataset Mutagenicity --arch GIN --seed 1
```

---

## 📊 Explanation Pipeline

1. Train or load a GNN model.
2. Extract graph/node activations from intermediate layers.
3. Train a Decision Tree surrogate model on graph embeddings.
4. Extract rules and predicates based on the surrogate.
5. Visualize subgraphs with orbit annotations and explanations.

---

## ✅ Outputs

The framework reports:

- Training and test accuracy of GNNs
- Surrogate decision tree fidelity vs. GNN predictions
- Weighted/unweighted precision, recall, and F1-scores
- Extracted rules and predicates with visualization of subgraphs