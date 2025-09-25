# 🧩 LogicGNN: Graph Neural Networks with Rule-Based Explanations

This repository provides an experimental framework for training Graph Neural Networks (GNNs) and extracting interpretable logic-based rules from their predictions. It combines standard GNN architectures with decision tree–based explanations and orbit-based grounding.

---

## 📌 Features
- Multiple GNN backbones:
  - **GCN**, **GIN**, **GAT**, **GraphSAGE**
- Wide range of datasets:
  - **Molecule datasets**: BBBP, Mutagenicity, NCI1  
  - **Synthetic**: BAMultiShapes  
  - **Social**: IMDB-BINARY, Reddit, Twitch, GitHub stargazers
- Explanation modules:
  - Extract intermediate activations from GNN layers
  - Fit surrogate **Decision Trees** for interpretability
  - Subgraph grounding with orbit decomposition and predicate extraction
- Training utilities:
  - Class-balanced weights
  - Checkpoint saving/loading
  - Early stopping thresholds per dataset

---

## 📂 Repository Structure

repo/
│── load_data.py # Dataset loaders and preprocessing
│── gnn.py # GCN, GIN, GAT, GraphSAGE implementations
│── main.py # Main training + explanation pipeline
│── utils.py # Training, testing, model checkpointing
│── explain_gnn.py # Activation extraction + decision tree explainer
│── grounding.py # Orbit decomposition + grounding predicates
│── models/ # Saved checkpoints (created at runtime)
│── plot/ # Generated plots (created at runtime)
|── readme/ # This file

---

## 🚀 Installation
Clone the repo and set up dependencies:

```bash
git clone https://github.com/your-username/logicgnn.git
cd logicgnn

conda create -n logicgnn python=3.10
conda activate logicgnn
pip install torch torch-geometric scikit-learn matplotlib networkx

## 🛠 Usage

### Training a Model
```bash
python main.py --dataset BBBP --arch GCN --seed 42
```
Arguments

--dataset: one of BBBP, Mutagenicity, IMDB-BINARY, NCI1, BAMultiShapes, reddit_threads, twitch_egos, github_stargazers

--arch: one of GCN, GIN, GAT, GraphSAGE

--seed: random seed (default: 0)

--load: load a pretrained model

--max_depth: maximum depth for decision tree explanations
Example

Train a GIN on Mutagenicity:
``` bash
python main.py --dataset Mutagenicity --arch GIN --seed 1
```
📊 Explanation Pipeline

Train or load a GNN model.

Extract graph/node activations from intermediate layers.

Train a Decision Tree surrogate model on graph embeddings.

Extract rules and predicates based on the surrogate.

Visualize subgraphs with orbit annotations and explanations.
✅ Outputs

The framework reports:

Training and test accuracy of GNNs

Surrogate decision tree fidelity vs. GNN predictions

Weighted/unweighted precision, recall, and F1-scores

Extracted rules and predicates with visualization of subgraphs