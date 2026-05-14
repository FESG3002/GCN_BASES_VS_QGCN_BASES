import torch
import torch.nn.functional as F
import time
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def train_mse_rating(model, data_or_adj, optimizer, num_users, num_items, 
                      train_users_t, train_items_t, train_ratings_t, device, use_pyg=True):
    model.train()
    optimizer.zero_grad()
    if use_pyg:
        all_embeddings = model(data_or_adj)
    else:
        all_embeddings = model(data_or_adj)

    pred_scores = model.predict_score(all_embeddings, train_users_t, train_items_t, num_users)
    normalized_ratings = 2 * (train_ratings_t - 1.0) / 4.0 - 1  
    loss = F.mse_loss(pred_scores, normalized_ratings)

    loss.backward()
    optimizer.step()
    return loss.item()

def train_bpr(model, data_or_adj, optimizer, num_users, num_items,
              train_users_t, train_items_t, device, use_pyg=True):
    model.train()
    optimizer.zero_grad()
    all_embeddings = model(data_or_adj)

    if use_pyg:
        pos_user_edge, pos_item_edge = data_or_adj.edge_index
        mask = pos_user_edge < num_users
        pos_user_edge = pos_user_edge[mask]
        pos_item_edge = pos_item_edge[mask]
        pos_scores = model.predict_score(all_embeddings, pos_user_edge, pos_item_edge - num_users, num_users)
        neg_items = torch.randint(0, num_items, (pos_user_edge.size(0),), dtype=torch.long, device=device)
        neg_scores = model.predict_score(all_embeddings, pos_user_edge, neg_items, num_users)
    else:
        pos_scores = model.predict_score(all_embeddings, train_users_t, train_items_t, num_users)
        neg_items = torch.randint(0, num_items, (len(train_users_t),), device=device)
        neg_scores = model.predict_score(all_embeddings, train_users_t, neg_items, num_users)

    diff = pos_scores - neg_scores
    loss = -torch.mean(torch.log(torch.sigmoid(diff) + 1e-10))

    if hasattr(model, 'embeddings'):
        reg_loss = 1e-6 * torch.norm(model.embeddings.weight, p=2)
        loss = loss + reg_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()

def train_quantum_hybrid(model, norm_adj_tensor, train_users, train_items,
                        num_users, num_items, n_epochs=100, batch_size=512,
                        lr=0.001, neg_samples=1, patience=15, device='cpu'):
    model = model.to(device)
    norm_adj_tensor = norm_adj_tensor.to(device)

    dataset = TensorDataset(
        torch.LongTensor(train_users),
        torch.LongTensor(train_items)
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []

        for u_batch, i_pos_batch in train_loader:
            u_batch = u_batch.to(device)
            i_pos_batch = i_pos_batch.to(device)

            optimizer.zero_grad()
            all_embeddings = model(norm_adj_tensor)

            batch_losses = []
            for _ in range(neg_samples):
                i_neg_batch = torch.randint(0, num_items, (len(u_batch),), device=device)

                pos_scores = model.predict_score(all_embeddings, u_batch, i_pos_batch)
                neg_scores = model.predict_score(all_embeddings, u_batch, i_neg_batch)

                pos_loss = F.mse_loss(pos_scores, torch.ones_like(pos_scores))
                neg_loss = F.mse_loss(neg_scores, torch.zeros_like(neg_scores))
                mse_loss = pos_loss + neg_loss
                batch_losses.append(mse_loss)

            loss = torch.mean(torch.stack(batch_losses))
            reg_loss = 1e-5 * (
                model.gcn.embeddings.weight[u_batch].norm(2) +
                model.gcn.embeddings.weight[i_pos_batch + num_users].norm(2)
            ) / len(u_batch)

            total_loss = loss + reg_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        scheduler.step(avg_loss)

        if avg_loss < best_loss - 1e-5:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    return model
