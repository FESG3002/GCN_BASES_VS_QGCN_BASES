import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score

@torch.no_grad()
def evaluate(model, adj_tensor, train_df, test_df, num_users, num_items, K_values, device='cpu'):
    print(f"Device utilisé : {device}")
    model.eval()

    all_embeddings = model(adj_tensor.to(device))

    test_user_inputs = torch.LongTensor(test_df['user_id_encoded'].values).to(device)
    test_item_inputs = torch.LongTensor(test_df['item_id_encoded'].values).to(device)
    true_ratings = torch.FloatTensor(test_df['rating'].values).to(device)

    # 1. PREDICTION METRICS
    if hasattr(model, 'predict_score') and 'predict_score' in dir(model):
        try:
            # GCNRecommender, etc
            predicted_scores = model.predict_score(all_embeddings, test_user_inputs, test_item_inputs, num_users)
        except TypeError:
            # QuantumHybridRecommender
            predicted_scores = model.predict_score(all_embeddings, test_user_inputs, test_item_inputs)
    else:
        # Fallback
        u_emb = all_embeddings[test_user_inputs]
        i_emb = all_embeddings[test_item_inputs + num_users]
        predicted_scores = (u_emb * i_emb).sum(dim=1)

    min_rating, max_rating = true_ratings.min(), true_ratings.max()
    scaled_predicted_ratings = torch.sigmoid(predicted_scores) * (max_rating - min_rating) + min_rating

    true_ratings_np = true_ratings.cpu().numpy()
    scaled_predicted_ratings_np = scaled_predicted_ratings.cpu().numpy()

    mse = mean_squared_error(true_ratings_np, scaled_predicted_ratings_np)
    mae = mean_absolute_error(true_ratings_np, scaled_predicted_ratings_np)
    rmse = np.sqrt(mse)

    # 2. PAIRWISE AUC
    pos_scores = predicted_scores
    neg_item_samples = torch.randint(0, num_items, (len(test_user_inputs),), dtype=torch.long, device=device)
    
    try:
        neg_scores = model.predict_score(all_embeddings, test_user_inputs, neg_item_samples, num_users)
    except TypeError:
        neg_scores = model.predict_score(all_embeddings, test_user_inputs, neg_item_samples)

    scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
    labels = torch.cat([torch.ones(pos_scores.shape[0]), torch.zeros(neg_scores.shape[0])]).cpu().numpy()
    auc_score = roc_auc_score(labels, scores)

    # 3. RANKING
    user_test_items = test_df.groupby('user_id_encoded')['item_id_encoded'].apply(set).to_dict()
    user_train_items = train_df.groupby('user_id_encoded')['item_id_encoded'].apply(set).to_dict()

    metrics = {k: {'hr': [], 'precision': [], 'recall': [], 'ndcg': []} for k in K_values}

    for user_id in user_test_items.keys():
        relevant_items = user_test_items[user_id]
        all_item_ids = torch.arange(num_items, device=device)
        user_tensor = torch.full((num_items,), user_id, dtype=torch.long, device=device)
        
        try:
            all_scores = model.predict_score(all_embeddings, user_tensor, all_item_ids, num_users)
        except TypeError:
            all_scores = model.predict_score(all_embeddings, user_tensor, all_item_ids)

        items_to_exclude = user_train_items.get(user_id, set())
        if items_to_exclude:
            all_scores[list(items_to_exclude)] = -float('inf')

        _, top_k_items = torch.topk(all_scores, k=max(K_values))
        top_k_items = top_k_items.cpu().numpy()

        for K in K_values:
            top_k_recommendations = top_k_items[:K]
            recommended_relevant = set(top_k_recommendations) & relevant_items
            num_recommended_relevant = len(recommended_relevant)

            metrics[K]['hr'].append(1.0 if num_recommended_relevant > 0 else 0.0)
            metrics[K]['precision'].append(num_recommended_relevant / K)
            metrics[K]['recall'].append(num_recommended_relevant / len(relevant_items))

            dcg, idcg = 0.0, 0.0
            for i, item in enumerate(top_k_recommendations):
                if item in relevant_items:
                    dcg += 1.0 / np.log2(i + 2)
            for i in range(min(len(relevant_items), K)):
                idcg += 1.0 / np.log2(i + 2)
            metrics[K]['ndcg'].append(dcg / idcg if idcg > 0 else 0.0)

    results = {
        'rating_prediction': {'MSE': mse, 'MAE': mae, 'RMSE': rmse},
        'pairwise_ranking': {'AUC': auc_score},
        'ranking_at_k': {}
    }
    for K in K_values:
        results['ranking_at_k'][f'@{K}'] = {
            'Hit_Rate': np.mean(metrics[K]['hr']),
            'Precision': np.mean(metrics[K]['precision']),
            'Recall': np.mean(metrics[K]['recall']),
            'NDCG': np.mean(metrics[K]['ndcg'])
        }

    return results

def print_evaluation_results(results):
    print("\n" + "="*70)
    print("📊 RÉSULTATS D'ÉVALUATION SUR LE TEST SET")
    print("="*70)

    print("\n🎯 MÉTRIQUES DE PRÉDICTION DE NOTES (POUR RÉFÉRENCE) :")
    print("--- (AVERTISSEMENT : Ces métriques sont peu fiables pour un modèle de classement) ---")
    rating_metrics = results['rating_prediction']
    print(f"  MSE  (Mean Squared Error)      : {rating_metrics['MSE']:.4f}")
    print(f"  MAE  (Mean Absolute Error)     : {rating_metrics['MAE']:.4f}")
    print(f"  RMSE (Root Mean Squared Error) : {rating_metrics['RMSE']:.4f}")

    print("\n🎯 MÉTRIQUE DE CLASSEMENT PAIRWISE  :")
    print("-" * 70)
    print(f"  AUC (Area Under Curve) : {results['pairwise_ranking']['AUC']:.4f}")

    print("\n📈 MÉTRIQUES DE CLASSEMENT TOP-K  :")
    print("-" * 70)
    for k_label, k_metrics in results['ranking_at_k'].items():
        print(f"\n  {k_label}:")
        print(f"    Hit Rate   : {k_metrics['Hit_Rate']:.4f} ({k_metrics['Hit_Rate']*100:.2f}%)")
        print(f"    Precision  : {k_metrics['Precision']:.4f}")
        print(f"    Recall     : {k_metrics['Recall']:.4f}")
        print(f"    NDCG       : {k_metrics['NDCG']:.4f}")
    print("\n" + "="*70)
