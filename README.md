# Quantum-Temporal GCN for Recommender Systems



## Purpose of the Code

The main purpose of this repository is the implementation and comparison of recommendation systems based on classical deep graph learning (Graph Convolutional Networks - GCN) and their quantum extensions (Quantum Machine Learning).

The project establishes two key concepts:
1. **Static and Temporal Learning (Time-Decay)**: Use of an adjacency matrix with temporal decay to model the fact that recent user-item interactions are more important than older ones.
2. **Quantum-Classical Hybridization**: Design of a model that leverages classical GCNs to capture the spatial topology of interactions and uses a quantum scorer (exploiting entanglement and superposition gates) to correct and model extraordinarily complex non-linear interactions.

---

## Models and Their References

### 1. Classical Models (GCN)
Graph learning algorithms exploit information propagation between users and movies (Items).

- **`GCNRecommender` (via PyTorch Geometric)**: The standard implementation.
  - *Subject*: Uses the classical GCNConv operator.
  - *Paper reference*: [Semi-Supervised Classification with Graph Convolutional Networks (Kipf & Welling, ICLR 2017)](https://arxiv.org/abs/1609.02907).
- **`GCNRecommenderFromScratch`**: The classical implementation without high-level modules.
  - *Subject*: Written from scratch (`CustomGCNLayer`) via the classical sparse matrix multiplication operation (`torch.sparse.mm`). Used as a backbone to remain fully transparent when fused with quantum neural networks (QNN).

*Note on the Loss Function:*
The models benefit from BPR recommendation optimization.
- *Paper reference*: [BPR: Bayesian Personalized Ranking from Implicit Feedback (Rendle et al., UAI 2009)](https://arxiv.org/abs/1205.2618).

### 2. Quantum and Hybrid Models (QML)
Quantum neural networks (QNN) or variational quantum circuits (VQC) introduce a different expressivity.

- **`QuantumEnhancedScorer` / `QuantumGCNLayer`**: Quantum module using Qiskit.
  - *Subject*: Uses parameterized quantum circuits (PQC) implementing rotations (`RY`, `RZ`) for encoding and control gates (`CX`) for entangling latent features. Integrated via `TorchConnector`.
  - *General QML reference*: [Classification with Quantum Neural Networks on Near Term Processors (Farhi & Neven, 2018)](https://arxiv.org/abs/1802.06002).
- **`QuantumHybridRecommender`**: The main contribution.
  - *Subject*: The model combines a classical GCN to learn the macroscopic graph architecture and projects the generated representation into a reduced-qubit quantum space. This system predicts affinity in the form of:
  `Score = Dot_Product(User, Item) + β * Quantum_Correction(User, Item)`.
  - *QGCN reference*: Inspired by the work of [Quantum Graph Neural Networks (Verdon et al., 2019)](https://arxiv.org/abs/1909.12264).

---

## System Structure

- `src/data_loader.py`: Imports data (MovieLens), encodes *IDs*, performs strict chronological splitting (Train/Val/Test), and generates weighted graphs (Static and Temporal with weighted *decay*).
- `src/models.py`: Abstract catalog listing our PyTorch classes for the precise definition of the algorithms described above (PyG, From Scratch, and Quantum Qiskit).
- `src/evaluation.py`: Offline evaluation logic for Recommender Systems (MSE/RMSE error calculations, AUC ranking metric, top-K metrics such as Recall@K and NDCG@K).
- `src/train.py`: Contains the BPR training algorithm for classical architectures, as well as `train_quantum_hybrid` optimized with gradient clipping, batch `DataLoader`, and very careful MSE/BPR loss handling for quantum models.
- `main.py`: Main *plug-and-play* file. It connects the end-to-end pipeline from data organization to comparative model evaluation.

## Execution Instructions

Ensure that the virtual environment includes all required libraries (`torch`, `torch_geometric`, `qiskit`, `qiskit-machine-learning`, `numpy`, `pandas`, `scipy`, `scikit-learn`).

```bash
# Launch the full standard experiment (GCN from scratch + Hybrid QGCN)
python main.py
```
