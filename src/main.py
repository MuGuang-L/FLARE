# src/main.py
"""训练KGFM模型的主脚本。"""

import argparse
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
from model.KGFM import KGFM
from utils import dataloader4kge, dataloader4KGNN, dataloader4Emb


# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train(
        epochs: int,
        batch_size: int,
        lr: float,
        dim: int,
        n_neighbors: int,
        eva_per_epochs: int
) -> None:
    """训练KGFM模型。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    users, items, train_set, test_set = dataloader4kge.readRecData()
    entitys, relations, kgTriples = dataloader4kge.readKGData()
    user_text, job_text = dataloader4Emb.getEmbedding()
    kg_indexes = dataloader4KGNN.getKgIndexsFromKgTriples(kgTriples)
    adj_entity, adj_relation = dataloader4KGNN.construct_adj(n_neighbors, kg_indexes, len(entitys))

    # 初始化模型
    model = KGFM(
        n_users=max(users) + 1,
        n_entities=max(entitys) + 1,
        n_relations=max(relations) + 1,
        dim=dim,
        adj_entity=adj_entity,
        adj_relation=adj_relation,
        user_text=user_text,
        job_text=job_text,
        drop_edge_rate=0.5,
        feat_drop_rate=0.5,
        device=device
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
    loss_fcn = torch.nn.BCEWithLogitsLoss()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 100

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for u, i, r in tqdm(train_loader, desc=f"第 {epoch + 1} 轮"):
            u, i, r = u.to(device), i.to(device), r.to(device)
            logits = model(u, i)
            loss = loss_fcn(logits, r.float())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step(total_loss)

        if epoch % eva_per_epochs == 0:
            train_metrics = evaluate(model, train_set, loss_fcn, batch_size, device)
            test_metrics = evaluate(model, test_set, loss_fcn, batch_size, device)
            logger.info(f"第 {epoch} 轮 | 训练损失: {train_metrics['loss']:.4f} | "
                        f"精确率: {train_metrics['precision']:.4f} | "
                        f"F1分数: {train_metrics['f1']:.4f}")
            logger.info(f"第 {epoch} 轮 | 测试损失: {test_metrics['loss']:.4f} | "
                        f"精确率: {test_metrics['precision']:.4f} | "
                        f"F1分数: {test_metrics['f1']:.4f}")

            if test_metrics['loss'] < best_val_loss:
                best_val_loss = test_metrics['loss']
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("提前停止触发。")
                    break


def evaluate(
        model: KGFM,
        dataset: list,
        loss_fcn: torch.nn.Module,
        batch_size: int,
        device: torch.device
) -> dict:
    """在数据集上评估模型。"""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for u, i, r in loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            logits = model(u, i)
            loss = loss_fcn(logits, r.float())
            total_loss += loss.item()
            y_true.extend(r.cpu().numpy())
            y_pred.extend((torch.sigmoid(logits) > 0.5).float().cpu().numpy())
            y_scores.extend(torch.sigmoid(logits).cpu().numpy())

    return {
        'loss': total_loss / len(loader),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_scores)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练KGFM模型")
    parser.add_argument("--epochs", type=int, default=500, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=128, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--dim", type=int, default=32, help="嵌入维度")
    parser.add_argument("--n_neighbors", type=int, default=5, help="邻居数量")
    parser.add_argument("--eva_per_epochs", type=int, default=1, help="评估频率")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dim=args.dim,
        n_neighbors=args.n_neighbors,
        eva_per_epochs=args.eva_per_epochs
    )