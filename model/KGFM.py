# src/model/kgfm.py
"""知识图谱分解机（KGFM）模型，用于推荐系统。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KGFM(nn.Module):
    """KGFM模型，结合知识图谱和分解机原理。"""

    def __init__(
            self,
            n_users: int,
            n_entities: int,
            n_relations: int,
            dim: int,
            adj_entity: torch.Tensor,
            adj_relation: torch.Tensor,
            user_text: torch.Tensor,
            job_text: torch.Tensor,
            n_layers: int = 1,
            agg_method: str = 'other',
            mp_method: str = 'FM_KGAT',
            drop_edge_rate: float = 0.4,
            feat_drop_rate: float = 0.4,
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        super().__init__()
        self.device = device
        self.n_layers = n_layers
        self.dim = dim
        self.agg_method = agg_method
        self.mp_method = mp_method
        self.adj_entity = adj_entity.to(device)
        self.adj_relation = adj_relation.to(device)
        self.drop_edge_rate = drop_edge_rate
        self.feat_drop_rate = feat_drop_rate

        # 初始化嵌入
        n_pretrained = user_text.size(0) + job_text.size(0)
        pretrained_embs = torch.cat([user_text, job_text], dim=0)
        random_embs = nn.Embedding(n_pretrained, dim, max_norm=1).weight.data
        random_embs = random_embs * pretrained_embs.std() + pretrained_embs.mean()
        full_embs = torch.cat([random_embs, pretrained_embs], dim=1)

        entity_embs = torch.zeros((n_entities, dim * 2))
        entity_embs[:n_pretrained] = full_embs
        if n_entities > n_pretrained:
            other_embs = nn.Embedding(n_entities - n_pretrained, dim * 2, max_norm=1).weight.data
            entity_embs[n_pretrained:] = other_embs

        self.entity_embs = nn.Embedding.from_pretrained(entity_embs, freeze=False)
        self.relation_embs = nn.Embedding(n_relations, dim * 2, max_norm=1)

        # 批归一化和线性层
        self.bn_entity = nn.BatchNorm1d(dim * 2)
        self.bn_relation = nn.BatchNorm1d(dim * 2)
        self.Wr1 = nn.Linear(dim * 2, dim * 2)
        self.w_last = nn.Linear(dim * 4, 1)

        self.leakyRelu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(0.4)

        if agg_method == 'concat':
            self.W_concat = nn.Linear(dim * 4, dim)
        else:
            self.W1 = nn.Linear(dim * 2, dim * 2)
            if agg_method == 'Bi-Interaction':
                self.W2 = nn.Linear(dim * 2, dim * 2)

    def get_neighbors(self, items: torch.Tensor) -> tuple:
        """获取邻居实体和关系的嵌入。"""
        items = items.to(self.device)
        e_ids = self.adj_entity[items]
        r_ids = self.adj_relation[items]
        neighbor_entities_embs = self.entity_embs(e_ids)
        neighbor_relations_embs = self.relation_embs(r_ids)

        if self.drop_edge_rate > 0 and self.training:
            edge_mask = (torch.rand_like(e_ids, dtype=torch.float, device=self.device) > self.drop_edge_rate).float()
            neighbor_entities_embs = neighbor_entities_embs * edge_mask.unsqueeze(-1)
            neighbor_relations_embs = neighbor_relations_embs * edge_mask.unsqueeze(-1)

        if self.feat_drop_rate > 0 and self.training:
            neighbor_entities_embs = F.dropout(neighbor_entities_embs, p=self.feat_drop_rate, training=True)
            neighbor_relations_embs = F.dropout(neighbor_relations_embs, p=self.feat_drop_rate, training=True)

        return neighbor_entities_embs, neighbor_relations_embs

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """前向传播，计算推荐分数。"""
        u, i = u.to(self.device), i.to(self.device)
        user_embs = self.bn_entity(self.entity_embs(u))
        item_embs = self.bn_entity(self.entity_embs(i))

        if self.feat_drop_rate > 0 and self.training:
            user_embs = F.dropout(user_embs, p=self.feat_drop_rate, training=True)
            item_embs = F.dropout(item_embs, p=self.feat_drop_rate, training=True)

        h_embs, u_embs = item_embs, user_embs
        for _ in range(self.n_layers):
            t_embs_item, r_embs_item = self.get_neighbors(i)
            Nh_embs_item = self.FMMessagePassFromKGAT(h_embs, r_embs_item, t_embs_item)
            out_item_embs = self.aggregate(h_embs, Nh_embs_item, self.agg_method)
            out_item_embs = self.dropout(out_item_embs)
            h_embs = h_embs + out_item_embs

            t_embs_user, r_embs_user = self.get_neighbors(u)
            Nh_embs_user = self.FMMessagePassFromKGAT(u_embs, r_embs_user, t_embs_user)
            out_user_embs = self.aggregate(u_embs, Nh_embs_user, self.agg_method)
            out_user_embs = self.dropout(out_user_embs)
            u_embs = u_embs + out_user_embs

        combined_features = torch.cat([u_embs, h_embs], dim=-1)
        return self.w_last(combined_features).squeeze(-1)

    def FMMessagePassFromKGAT(self, h_embs: torch.Tensor, r_embs: torch.Tensor, t_embs: torch.Tensor) -> torch.Tensor:
        """基于FM和KGAT的消息传递。"""
        h_broadcast_embs = h_embs.unsqueeze(1).expand(-1, t_embs.size(1), -1)
        tr_embs = self.dropout(self.Wr1(t_embs))
        hr_embs = self.dropout(self.Wr1(h_broadcast_embs))
        hr_embs = torch.tanh(hr_embs + r_embs)
        hrt_embs = hr_embs * tr_embs
        hrt_embs = self.dropout(hrt_embs)
        square_of_sum = torch.sum(hrt_embs, dim=1) ** 2
        sum_of_square = torch.sum(hrt_embs ** 2, dim=1)
        return square_of_sum - sum_of_square

    def aggregate(self, h_embs: torch.Tensor, Nh_embs: torch.Tensor, agg_method: str) -> torch.Tensor:
        """聚合邻居信息。"""
        if agg_method == 'Bi-Interaction':
            return self.leakyRelu(self.W1(h_embs + Nh_embs)) + self.leakyRelu(self.W2(h_embs * Nh_embs))
        elif agg_method == 'concat':
            return self.leakyRelu(self.W_concat(torch.cat([h_embs, Nh_embs], dim=-1)))
        return self.leakyRelu(self.W1(h_embs + Nh_embs))