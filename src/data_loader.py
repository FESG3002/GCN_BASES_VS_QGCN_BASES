import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

def load_and_preprocess_data(sample_size=None, train_ratio=0.70, val_ratio=0.10, test_ratio=0.20):
    print("\n[1/3] Chargement des données MovieLens...")
    url = 'http://files.grouplens.org/datasets/movielens/ml-100k/u.data'
    df = pd.read_csv(url, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
    
    if sample_size is not None:
        df = df[:sample_size]
        
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    df['user_id_encoded'] = user_encoder.fit_transform(df['user_id'])
    df['item_id_encoded'] = item_encoder.fit_transform(df['item_id'])
    
    num_users = df['user_id_encoded'].nunique()
    num_items = df['item_id_encoded'].nunique()
    num_nodes = num_users + num_items
    
    df_sorted = df.sort_values(by='timestamp').reset_index(drop=True)
    train_end_idx = int(len(df_sorted) * train_ratio)
    val_end_idx = int(len(df_sorted) * (train_ratio + val_ratio))
    
    train_df = df_sorted.iloc[:train_end_idx]
    val_df = df_sorted.iloc[train_end_idx:val_end_idx]
    test_df = df_sorted.iloc[val_end_idx:]
    
    print(f"✓ Total interactions : {len(df):,}")
    print(f"✓ Utilisateurs : {num_users}, Items : {num_items}, Noeuds : {num_nodes}")
    print(f"✓ Train : {len(train_df):,} | Val : {len(val_df):,} | Test : {len(test_df):,}")
    
    return df, train_df, val_df, test_df, num_users, num_items, num_nodes, user_encoder, item_encoder

def build_pyg_data(train_df, num_users, num_nodes):
    user_nodes = torch.tensor(train_df['user_id_encoded'].values, dtype=torch.long)
    item_nodes = torch.tensor(train_df['item_id_encoded'].values, dtype=torch.long) + num_users
    
    edge_index = torch.stack([user_nodes, item_nodes], dim=0)
    edge_index = to_undirected(edge_index)
    
    ratings_tensor = torch.tensor(train_df['rating'].values, dtype=torch.float)
    if edge_index.shape[1] > len(ratings_tensor):
        ratings_tensor = torch.cat([ratings_tensor, ratings_tensor])
        
    x = torch.eye(num_nodes, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=ratings_tensor)
    return data

def build_weighted_adjacency_matrix(users, items, weights, num_nodes, num_users, use_weights=True):
    adj_matrix = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32)
    item_ids_shifted = items + num_users

    if use_weights:
        normalized_weights = (weights - weights.min()) / (weights.max() - weights.min())
        adj_matrix[users, item_ids_shifted] = normalized_weights
        adj_matrix[item_ids_shifted, users] = normalized_weights
    else:
        adj_matrix[users, item_ids_shifted] = 1.0
        adj_matrix[item_ids_shifted, users] = 1.0

    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    A_hat = adj_matrix + sp.eye(num_nodes, dtype=np.float32)

    row_sum = np.array(A_hat.sum(1)).flatten() + 1e-10
    d_inv_sqrt = np.power(row_sum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = D_inv_sqrt @ A_hat @ D_inv_sqrt

    norm_adj_coo = norm_adj.tocoo()
    indices = torch.LongTensor(np.vstack((norm_adj_coo.row, norm_adj_coo.col)))
    values = torch.FloatTensor(norm_adj_coo.data)
    shape = torch.Size(norm_adj_coo.shape)

    return torch.sparse_coo_tensor(indices, values, shape)

def build_weighted_adjacency_matrix_with_decay(users, items, ratings, timestamps,
                                                num_nodes, num_users,
                                                use_ratings=True, decay_rate=0.01,
                                                reference_timestamp=None):
    if reference_timestamp is None:
        reference_timestamp = np.max(timestamps)

    time_diffs = (reference_timestamp - timestamps) / (3600 * 24)
    temporal_decay_factors = np.exp(-decay_rate * time_diffs)
    temporal_decay_factors = np.clip(temporal_decay_factors, a_min=1e-5, a_max=1.0)

    adj_matrix = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32)
    item_ids_shifted = items + num_users

    if use_ratings:
        min_rating = np.min(ratings)
        max_rating = np.max(ratings)
        normalized_ratings = (ratings - min_rating) / (max_rating - min_rating) if max_rating > min_rating else np.ones_like(ratings)
        final_weights = normalized_ratings * temporal_decay_factors
    else:
        final_weights = temporal_decay_factors

    rows_coo = np.concatenate([users, item_ids_shifted])
    cols_coo = np.concatenate([item_ids_shifted, users])
    data_coo = np.concatenate([final_weights, final_weights])

    adj_matrix_coo = sp.coo_matrix((data_coo, (rows_coo, cols_coo)), shape=(num_nodes, num_nodes), dtype=np.float32)
    adj_matrix = adj_matrix_coo.tocsr()

    A_hat = adj_matrix + sp.eye(num_nodes, dtype=np.float32)

    row_sum = np.array(A_hat.sum(1)).flatten() + 1e-10
    d_inv_sqrt = np.power(row_sum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_adj = d_mat_inv_sqrt.dot(A_hat).dot(d_mat_inv_sqrt)

    coo = normalized_adj.tocoo()
    indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
    values = torch.FloatTensor(coo.data)
    shape = torch.Size(coo.shape)

    return torch.sparse_coo_tensor(indices, values, shape).coalesce()
