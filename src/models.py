import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives.statevector_estimator import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

class GCNRecommender(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super(GCNRecommender, self).__init__()
        self.conv1 = GCNConv(num_nodes, 128)
        self.conv2 = GCNConv(128, embedding_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def predict_score(self, embeddings, user_id, item_id, num_users):
        user_emb = embeddings[user_id]
        item_emb = embeddings[item_id + num_users]
        score = (user_emb * item_emb).sum(dim=1)
        return score

class CustomGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomGCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, node_features, adj_matrix):
        aggregated_features = torch.sparse.mm(adj_matrix, node_features)
        output_features = self.linear(aggregated_features)
        return output_features

class GCNRecommenderFromScratch(nn.Module):
    def __init__(self, num_nodes, embedding_dim, gcn_hidden_dim=128, final_embedding_dim=64):
        super(GCNRecommenderFromScratch, self).__init__()
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.embeddings.weight)
        self.gcn_layer1 = CustomGCNLayer(embedding_dim, gcn_hidden_dim)
        self.gcn_layer2 = CustomGCNLayer(gcn_hidden_dim, final_embedding_dim)

    def forward(self, adj_matrix):
        initial_features = self.embeddings.weight
        x = self.gcn_layer1(initial_features, adj_matrix)
        x = F.relu(x)
        final_embeddings = self.gcn_layer2(x, adj_matrix)
        return final_embeddings

    def predict_score(self, embeddings, user_indices, item_indices, num_users):
        user_embs = embeddings[user_indices]
        item_embs = embeddings[item_indices + num_users]
        scores = (user_embs * item_embs).sum(dim=1)
        return scores

class QuantumGCNLayer(nn.Module):
    def __init__(self, in_features, out_features, n_layers=2):
        super().__init__()
        assert in_features == out_features, "Input and output features must be equal"
        self.n_qubits = in_features
        self.n_layers = n_layers

        inputs = ParameterVector("input", self.n_qubits)
        weights = ParameterVector("weight", self.n_qubits * n_layers * 2)

        qc = QuantumCircuit(self.n_qubits)

        for i in range(self.n_qubits):
            qc.ry(inputs[i], i)
            qc.rz(inputs[i], i)

        weight_idx = 0
        for layer in range(n_layers):
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
            if self.n_qubits > 2:
                qc.cx(self.n_qubits - 1, 0)

            for i in range(self.n_qubits):
                qc.ry(weights[weight_idx], i)
                weight_idx += 1
                qc.rz(weights[weight_idx], i)
                weight_idx += 1

        observables = []
        for i in range(self.n_qubits):
            observables.append(SparsePauliOp("I" * i + "Z" + "I" * (self.n_qubits - 1 - i)))

        self.qnn = EstimatorQNN(
            circuit=qc,
            input_params=inputs,
            weight_params=weights,
            observables=observables,
            estimator=StatevectorEstimator(),
        )

        n_weights = self.n_qubits * n_layers * 2
        self.quantum_weights = nn.Parameter(
            torch.randn(n_weights) * np.pi / 4
        )

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x

        weights_np = self.quantum_weights.detach().cpu().numpy()
        output = self.qnn.forward(x_np, weights_np)
        return torch.tensor(output, dtype=torch.float32, requires_grad=True).to(x.device)

class HybridGCNRecommender(nn.Module):
    def __init__(self, num_nodes, n_qubits, n_quantum_layers=2, n_gcn_layers=2):
        super().__init__()
        self.n_gcn_layers = n_gcn_layers

        self.embeddings = nn.Embedding(num_nodes, n_qubits)
        nn.init.normal_(self.embeddings.weight, mean=0, std=np.pi/4)

        self.quantum_layers = nn.ModuleList([
            QuantumGCNLayer(n_qubits, n_qubits, n_layers=n_quantum_layers)
            for _ in range(n_gcn_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(n_qubits) for _ in range(n_gcn_layers)
        ])

        self.skip_weights = nn.Parameter(torch.ones(n_gcn_layers))

    def forward(self, adj_matrix):
        x = self.embeddings.weight

        for i, (quantum_layer, norm) in enumerate(zip(self.quantum_layers, self.layer_norms)):
            residual = x
            x = torch.sparse.mm(adj_matrix, x)
            x = quantum_layer(x)
            x = norm(x)
            x = x + self.skip_weights[i] * residual

        return x

    def predict_score(self, embeddings, user_indices, item_indices, num_users):
        user_embs = embeddings[user_indices]
        item_embs = embeddings[item_indices + num_users]
        scores = (user_embs * item_embs).sum(dim=1)
        return scores

class QuantumEnhancedScorer(nn.Module):
    def __init__(self, embedding_dim, n_qubits=2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_qubits = n_qubits

        self.to_quantum = nn.Linear(embedding_dim, n_qubits, bias=False)
        nn.init.orthogonal_(self.to_quantum.weight, gain=0.5)

        params = ParameterVector("features", n_qubits * 2)
        qc = QuantumCircuit(n_qubits)

        for i in range(n_qubits):
            qc.h(i)
            qc.ry(params[i], i)

        if n_qubits > 1:
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)

        for i in range(n_qubits):
            qc.ry(params[n_qubits + i], i)

        if n_qubits > 1:
            qc.cx(0, n_qubits - 1)

        pauli_string = "Z" * n_qubits
        observable = SparsePauliOp.from_list([(pauli_string, 1.0 / n_qubits)])

        qnn = EstimatorQNN(
            circuit=qc,
            input_params=list(params),
            observables=[observable]
        )

        self.quantum_module = TorchConnector(qnn)
        self.beta = nn.Parameter(torch.tensor(0.05))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u_emb, i_emb):
        u_norm = F.normalize(u_emb, p=2, dim=1)
        i_norm = F.normalize(i_emb, p=2, dim=1)
        dot_product = torch.sum(u_norm * i_norm, dim=1)

        u_quantum = torch.tanh(self.to_quantum(u_emb))
        i_quantum = torch.tanh(self.to_quantum(i_emb))

        u_quantum = F.normalize(u_quantum, p=2, dim=1) * (np.pi / 4)
        i_quantum = F.normalize(i_quantum, p=2, dim=1) * (np.pi / 4)

        quantum_input = torch.cat([u_quantum, i_quantum], dim=1)
        quantum_correction = self.quantum_module(quantum_input).squeeze()

        hybrid_score = dot_product + self.beta * quantum_correction + self.bias
        final_score = torch.sigmoid(hybrid_score)
        return final_score

class QuantumHybridRecommender(nn.Module):
    def __init__(self, num_nodes, num_users, initial_embedding_dim=64, n_qubits=2,
                 gcn_hidden_dim=128, final_embedding_dim=64):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_users = num_users
        self.embedding_dim = final_embedding_dim

        self.gcn = GCNRecommenderFromScratch(
            num_nodes=num_nodes,
            embedding_dim=initial_embedding_dim,
            gcn_hidden_dim=gcn_hidden_dim,
            final_embedding_dim=final_embedding_dim
        )
        self.scorer = QuantumEnhancedScorer(self.embedding_dim, n_qubits)

    def forward(self, adj_matrix):
        return self.gcn(adj_matrix)

    def predict_score(self, embeddings, user_indices, item_indices):
        u_emb = embeddings[user_indices]
        i_emb = embeddings[item_indices + self.num_users]
        return self.scorer(u_emb, i_emb)
