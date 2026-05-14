import torch
import numpy as np

from src.data_loader import (load_and_preprocess_data, build_pyg_data, 
                             build_weighted_adjacency_matrix_with_decay)
from src.models import (GCNRecommender, GCNRecommenderFromScratch, HybridGCNRecommender, QuantumHybridRecommender)
from src.evaluation import evaluate, print_evaluation_results
from src.train import train_bpr, train_quantum_hybrid

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 1. Charger et préparer les données
    df, train_df, val_df, test_df, num_users, num_items, num_nodes, user_enc, item_enc = load_and_preprocess_data(
        sample_size=None, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2
    )

    train_users = train_df['user_id_encoded'].values
    train_items = train_df['item_id_encoded'].values
    train_ratings = train_df['rating'].values
    train_timestamps = train_df['timestamp'].values

    # 2. Matrices d'adjacence
    norm_adj_tensor = build_weighted_adjacency_matrix_with_decay(
        train_users, train_items, train_ratings, train_timestamps,
        num_nodes, num_users, use_ratings=True, decay_rate=0.01
    )
    norm_adj_tensor = norm_adj_tensor.to(device)

    # Variables d'évaluation
    K_values = [10, 30, 50]

    # --- Exemple : GCN de Zéro (Classique) ---
    print("\n--- GCN Classique (from scratch) ---")
    model_scratch = GCNRecommenderFromScratch(num_nodes=num_nodes, embedding_dim=64).to(device)
    optimizer = torch.optim.Adam(model_scratch.parameters(), lr=0.01)

    print("Entraînement avec BPR Loss (10 epochs de démo)...")
    train_users_t = torch.LongTensor(train_users).to(device)
    train_items_t = torch.LongTensor(train_items).to(device)

    for epoch in range(10):
        loss = train_bpr(model_scratch, norm_adj_tensor, optimizer, num_users, num_items,
                         train_users_t, train_items_t, device, use_pyg=False)
        print(f"Epoch {epoch+1} | Loss BPR: {loss:.4f}")

    print("Évaluation modèle Classique...")
    results = evaluate(model_scratch, norm_adj_tensor, train_df, test_df, num_users, num_items, K_values, device)
    print_evaluation_results(results)


    # --- Exemple : Quantum Hybrid Recommender ---
    print("\n--- Modèle Hybride Quantique ---")
    model_quantum = QuantumHybridRecommender(
        num_nodes=num_nodes, num_users=num_users, initial_embedding_dim=64, 
        n_qubits=2, gcn_hidden_dim=128, final_embedding_dim=64
    ).to(device)

    print("Entraînement Quantum Hybrid (3 epochs de démo)...")
    trained_model = train_quantum_hybrid(
        model=model_quantum, norm_adj_tensor=norm_adj_tensor, train_users=train_users,
        train_items=train_items, num_users=num_users, num_items=num_items,
        n_epochs=3, batch_size=512, lr=0.01, neg_samples=1, patience=2, device=device
    )

    print("Évaluation modèle Quantique...")
    final_results = evaluate(trained_model, norm_adj_tensor, train_df, test_df, num_users, num_items, K_values, device)
    print_evaluation_results(final_results)

if __name__ == '__main__':
    main()
