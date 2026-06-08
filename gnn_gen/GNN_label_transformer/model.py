from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv

from transformer_decoder import EdgeDecoderConfig, build_edge_decoder


@dataclass
class ModelConfig:
    node_in: int
    num_classes: int
    hidden: int = 64
    layers: int = 3
    edge_hidden: int = 64
    self_eps: float = 1.0
    molecule_balanced_pool: bool = True

    use_branch_feature: bool = True
    branch_feature_mode: str = "scalar"  # "scalar" or "contextual"
    max_branch_slots: int = 8
    branch_emb_dim: int = 8
    branch_context_dim: int = 16

    use_third_body_feature: bool = True

    # Optional CVAE-style latent branch variable.
    # When enabled, the decoder receives a learned latent z per reactant-group.
    use_latent_branching: bool = False
    latent_dim: int = 8
    latent_hidden: int = 64

    # Optional auxiliary regression head for reaction-rate coefficients.
    predict_rate_coeffs: bool = False
    rate_out_dim: int = 6
    rate_hidden: int = 64

    # Transformer encoder inserted after GINEConv layers.
    use_transformer: bool = False
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_dropout: float = 0.1

    # Edge decoder selection.
    decoder_type: str = "mlp"  # "mlp" or "transformer"
    decoder_transformer_heads: int = 4
    decoder_transformer_layers: int = 2
    decoder_transformer_dropout: float = 0.1
    decoder_max_groups: int = 32


class EdgePredictor(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.branch_feature_mode not in {"scalar", "contextual"}:
            raise ValueError("branch_feature_mode must be 'scalar' or 'contextual'")

        self.convs = nn.ModuleList()
        for i in range(cfg.layers):
            mlp = nn.Sequential(
                nn.Linear(cfg.node_in if i == 0 else cfg.hidden, cfg.hidden),
                nn.ReLU(),
                nn.Linear(cfg.hidden, cfg.hidden),
            )
            self.convs.append(GINEConv(mlp, edge_dim=1, eps=cfg.self_eps, train_eps=False))

        # Optional learned categorical branch embedding contextualized by reactant.
        if cfg.use_branch_feature and cfg.branch_feature_mode == "contextual":
            self.branch_emb = nn.Embedding(cfg.max_branch_slots, cfg.branch_emb_dim)
            self.branch_fuse = nn.Sequential(
                nn.Linear(cfg.hidden + cfg.branch_emb_dim, cfg.branch_context_dim),
                nn.ReLU(),
                nn.Linear(cfg.branch_context_dim, cfg.branch_context_dim),
            )

        if cfg.use_transformer:
            if cfg.hidden % cfg.transformer_heads != 0:
                raise ValueError(
                    f"hidden ({cfg.hidden}) must be divisible by transformer_heads ({cfg.transformer_heads})"
                )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.hidden,
                nhead=cfg.transformer_heads,
                dim_feedforward=cfg.hidden * 4,
                dropout=cfg.transformer_dropout,
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.transformer_layers)

        if cfg.use_latent_branching:
            self.prior_net = nn.Sequential(
                nn.Linear(cfg.hidden, cfg.latent_hidden),
                nn.ReLU(),
                nn.Linear(cfg.latent_hidden, 2 * cfg.latent_dim),
            )
            self.posterior_net = nn.Sequential(
                nn.Linear(cfg.hidden + cfg.num_classes, cfg.latent_hidden),
                nn.ReLU(),
                nn.Linear(cfg.latent_hidden, 2 * cfg.latent_dim),
            )

        in_dim = cfg.hidden * 2 + 1
        if cfg.molecule_balanced_pool:
            in_dim += cfg.hidden

        if cfg.use_branch_feature:
            if cfg.branch_feature_mode == "scalar":
                in_dim += 1
            else:
                in_dim += cfg.branch_context_dim

        if cfg.use_third_body_feature:
            in_dim += 1

        if cfg.use_latent_branching:
            in_dim += cfg.latent_dim

        self.edge_decoder = build_edge_decoder(
            EdgeDecoderConfig(
                input_dim=in_dim,
                num_classes=cfg.num_classes,
                decoder_type=cfg.decoder_type,
                edge_hidden=cfg.edge_hidden,
                transformer_heads=cfg.decoder_transformer_heads,
                transformer_layers=cfg.decoder_transformer_layers,
                transformer_dropout=cfg.decoder_transformer_dropout,
                max_groups=cfg.decoder_max_groups,
            )
        )
# regression head, input: group pooled feature
        if cfg.predict_rate_coeffs:
            rate_in_dim = cfg.hidden
            if cfg.use_branch_feature:
                if cfg.branch_feature_mode == "scalar":
                    rate_in_dim += 1
                else:
                    rate_in_dim += cfg.branch_context_dim
            if cfg.use_third_body_feature:
                rate_in_dim += 1
            if cfg.use_latent_branching:
                rate_in_dim += cfg.latent_dim

            self.rate_mlp = nn.Sequential(
                nn.Linear(rate_in_dim, cfg.rate_hidden),
                nn.ReLU(),
                nn.Linear(cfg.rate_hidden, cfg.rate_out_dim),
            )

    def encode(self, data: Data) -> torch.Tensor:
        x = data.x
        for conv in self.convs:
            x = conv(x, data.edge_index, data.edge_attr)
            x = F.relu(x)
        if self.cfg.use_transformer:
            # TransformerEncoder expects [batch, seq, dim]; here batch=1, seq=N atoms.
            x = self.transformer_encoder(x.unsqueeze(0)).squeeze(0)
        return x

    def _molecule_balanced_pool(self, x: torch.Tensor, data: Data) -> torch.Tensor:
        if not hasattr(data, "mol_id"):
            return x.mean(dim=0)
        mol_id = data.mol_id
        if mol_id.numel() == 0:
            return x.mean(dim=0)
        num_mol = int(mol_id.max().item()) + 1
        pooled: List[torch.Tensor] = []
        for m in range(num_mol):
            mask = mol_id == m
            if torch.any(mask):
                pooled.append(x[mask].mean(dim=0))
        if not pooled:
            return x.mean(dim=0)
        return torch.stack(pooled, dim=0).sum(dim=0)

    def _global_scalar_feature(self, data: Data, field_name: str, num_edges: int, device: torch.device) -> torch.Tensor:
        if not hasattr(data, field_name):
            return torch.zeros((num_edges, 1), dtype=torch.float32, device=device)
        v = getattr(data, field_name).float()
        if v.dim() == 0:
            v = v.view(1)
        if v.numel() == 1:
            return v.view(1, 1).to(device).expand(num_edges, 1)
        # Batch size > 1 is not expected in this project; fall back safely.
        return v.view(-1, 1)[0:1].to(device).expand(num_edges, 1)

    def _edge_group_ids(self, data: Data, num_edges: int, device: torch.device) -> torch.Tensor:
        if hasattr(data, "pair_group_id"):
            g = data.pair_group_id.long().view(-1)
            if g.numel() == num_edges:
                return g.to(device)
        return torch.zeros(num_edges, dtype=torch.long, device=device)

    def _node_group_ids(self, data: Data, num_nodes: int, device: torch.device) -> torch.Tensor:
        if hasattr(data, "node_group_id"):
            g = data.node_group_id.long().view(-1)
            if g.numel() == num_nodes:
                return g.to(device)
        return torch.zeros(num_nodes, dtype=torch.long, device=device)

    def _num_groups(self, node_group: torch.Tensor, edge_group: torch.Tensor, group_attr_len: int) -> int:
        max_node = int(node_group.max().item()) if node_group.numel() > 0 else -1
        max_edge = int(edge_group.max().item()) if edge_group.numel() > 0 else -1
        max_attr = group_attr_len - 1
        return max(1, max_node + 1, max_edge + 1, max_attr + 1)

    def _pooled_group_context(
        self,
        x: torch.Tensor, # all the embedded node features, shape [N, H]
        data: Data, 
        node_group: torch.Tensor,
        num_groups: int,
    ) -> torch.Tensor:
        group_ctx = []
        mol_id = data.mol_id.long() if hasattr(data, "mol_id") else None

        for g in range(num_groups):
            g_mask = node_group == g
            if not torch.any(g_mask):
                group_ctx.append(torch.zeros(x.size(1), dtype=x.dtype, device=x.device))
                continue

            if self.cfg.molecule_balanced_pool and mol_id is not None and mol_id.numel() == x.size(0):
                group_mols = torch.unique(mol_id[g_mask], sorted=True)
                pooled = []
                for m in group_mols.tolist():
                    mask = g_mask & (mol_id == m)
                    if torch.any(mask):
                        pooled.append(x[mask].mean(dim=0))
                if pooled:
                    group_ctx.append(torch.stack(pooled, dim=0).sum(dim=0))
                else:
                    group_ctx.append(x[g_mask].mean(dim=0))
            else:
                group_ctx.append(x[g_mask].mean(dim=0))

        return torch.stack(group_ctx, dim=0)

    def _edge_group_pooled_feature(
        self,
        x: torch.Tensor,
        data: Data,
        edge_group: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns a per-edge pooled context tensor [E, H] gathered by edge group id.
        With single-group data this is equivalent to the previous global pooled vector.
        """
        num_edges = edge_group.numel()
        if num_edges == 0:
            return torch.zeros((0, x.size(1)), dtype=x.dtype, device=x.device)

        node_group = self._node_group_ids(data, x.size(0), x.device)
        num_groups = self._num_groups(
            node_group=node_group,
            edge_group=edge_group,
            group_attr_len=0,
        )
        pooled_by_group = self._pooled_group_context(
            x=x,
            data=data,
            node_group=node_group,
            num_groups=num_groups,
        )
        gather_idx = torch.clamp(edge_group, min=0, max=num_groups - 1)
        return pooled_by_group[gather_idx]

    def _edge_group_scalar_feature(
        self,
        data: Data,
        edge_group: torch.Tensor,
        group_field_name: str,
        legacy_field_name: str,
        device: torch.device,
    ) -> torch.Tensor:
        if hasattr(data, group_field_name):
            vals = getattr(data, group_field_name).float().view(-1).to(device)
        elif hasattr(data, legacy_field_name):
            vals = getattr(data, legacy_field_name).float().view(-1).to(device)
        else:
            vals = torch.zeros(1, dtype=torch.float32, device=device)

        if vals.numel() == 0:
            vals = torch.zeros(1, dtype=torch.float32, device=device)
        if vals.numel() == 1:
            return vals.view(1, 1).expand(edge_group.numel(), 1)

        idx = torch.clamp(edge_group, min=0, max=vals.numel() - 1)
        return vals[idx].view(-1, 1)

    def _group_scalar_feature(
        self,
        data: Data,
        num_groups: int,
        group_field_name: str,
        legacy_field_name: str,
        device: torch.device,
    ) -> torch.Tensor:
        if hasattr(data, group_field_name):
            vals = getattr(data, group_field_name).float().view(-1).to(device)
        elif hasattr(data, legacy_field_name):
            vals = getattr(data, legacy_field_name).float().view(-1).to(device)
        else:
            vals = torch.zeros(1, dtype=torch.float32, device=device)

        if vals.numel() == 0:
            vals = torch.zeros(1, dtype=torch.float32, device=device)
        if vals.numel() == 1:
            return vals.view(1, 1).expand(num_groups, 1)

        if vals.numel() < num_groups:
            pad = vals[-1:].expand(num_groups - vals.numel())
            vals = torch.cat([vals, pad], dim=0)
        return vals[:num_groups].view(-1, 1)

    def _contextual_branch_feature_by_group(
        self,
        x: torch.Tensor,
        data: Data,
        node_group: torch.Tensor,
        num_groups: int,
        device: torch.device,
    ) -> torch.Tensor:
        if hasattr(data, "branch_id_by_group"):
            b_raw = data.branch_id_by_group.long().view(-1).to(device)
        elif hasattr(data, "branch_id"):
            b_raw = data.branch_id.long().view(-1).to(device)
        else:
            b_raw = torch.zeros(1, dtype=torch.long, device=device)

        if b_raw.numel() < num_groups:
            pad = torch.zeros(num_groups - b_raw.numel(), dtype=torch.long, device=device)
            b_raw = torch.cat([b_raw, pad], dim=0)
        b_idx = torch.clamp(b_raw[:num_groups], min=0, max=self.cfg.max_branch_slots - 1)

        z_by_group = self._pooled_group_context(x=x, data=data, node_group=node_group, num_groups=num_groups)
        b_emb = self.branch_emb(b_idx)  # [G, branch_emb_dim]
        cond_in = torch.cat([z_by_group, b_emb], dim=-1)
        return self.branch_fuse(cond_in)  # [G, branch_context_dim]

    def _contextual_branch_feature(
        self,
        x: torch.Tensor,
        data: Data,
        edge_group: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        num_edges = edge_group.numel()
        if num_edges == 0:
            return torch.zeros((0, self.cfg.branch_context_dim), dtype=torch.float32, device=device)

        node_group = self._node_group_ids(data, x.size(0), device)

        if hasattr(data, "branch_id_by_group"):
            group_attr_len = data.branch_id_by_group.numel()
        elif hasattr(data, "branch_id"):
            group_attr_len = data.branch_id.numel()
        else:
            group_attr_len = 1

        num_groups = self._num_groups(node_group=node_group, edge_group=edge_group, group_attr_len=group_attr_len)
        cond_by_group = self._contextual_branch_feature_by_group(
            x=x,
            data=data,
            node_group=node_group,
            num_groups=num_groups,
            device=device,
        )

        gather_idx = torch.clamp(edge_group, min=0, max=num_groups - 1)
        return cond_by_group[gather_idx]

    def _latent_group_target_stats(
        self,
        data: Data,
        edge_group: torch.Tensor,
        num_groups: int,
        device: torch.device,
    ) -> torch.Tensor:
        stats = torch.zeros((num_groups, self.cfg.num_classes), dtype=torch.float32, device=device)
        if not hasattr(data, "y_edge"):
            return stats

        y = data.y_edge.long().view(-1).to(device)
        if y.numel() == 0 or edge_group.numel() == 0:
            return stats

        # Defensive clipping in case of data corruption.
        y = torch.clamp(y, min=0, max=self.cfg.num_classes - 1)
        onehot = F.one_hot(y, num_classes=self.cfg.num_classes).float()

        for g in range(num_groups):
            mask = edge_group == g
            if torch.any(mask):
                stats[g] = onehot[mask].mean(dim=0)
        return stats

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _kl_diag_gaussians(
        mu_q: torch.Tensor,
        logvar_q: torch.Tensor,
        mu_p: torch.Tensor,
        logvar_p: torch.Tensor,
    ) -> torch.Tensor:
        # KL(q||p) for diagonal Gaussians; returns scalar mean over groups.
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        kl = 0.5 * (
            (logvar_p - logvar_q)
            + (var_q + (mu_q - mu_p) ** 2) / torch.clamp(var_p, min=1e-12)
            - 1.0
        )
        return kl.sum(dim=-1).mean()

    def _latent_edge_feature(
        self,
        x: torch.Tensor,
        data: Data,
        edge_group: torch.Tensor,
        sample_latent: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x.device
        num_edges = edge_group.numel()

        node_group = self._node_group_ids(data, x.size(0), device)
        num_groups = self._num_groups(node_group=node_group, edge_group=edge_group, group_attr_len=0)
        react_ctx_by_group = self._pooled_group_context(
            x=x,
            data=data,
            node_group=node_group,
            num_groups=num_groups,
        )

        prior_params = self.prior_net(react_ctx_by_group)
        mu_p, logvar_p = torch.chunk(prior_params, 2, dim=-1)

        if self.training and hasattr(data, "y_edge"):
            y_stats = self._latent_group_target_stats(
                data=data,
                edge_group=edge_group,
                num_groups=num_groups,
                device=device,
            )
            post_in = torch.cat([react_ctx_by_group, y_stats], dim=-1)
            post_params = self.posterior_net(post_in)
            mu_q, logvar_q = torch.chunk(post_params, 2, dim=-1)
            z_group = self._reparameterize(mu_q, logvar_q, sample=sample_latent)
            kl_loss = self._kl_diag_gaussians(mu_q, logvar_q, mu_p, logvar_p)
        else:
            z_group = self._reparameterize(mu_p, logvar_p, sample=sample_latent)
            kl_loss = torch.zeros((), dtype=torch.float32, device=device)

        if num_edges == 0:
            z_edge = torch.zeros((0, self.cfg.latent_dim), dtype=torch.float32, device=device)
        else:
            gather_idx = torch.clamp(edge_group, min=0, max=num_groups - 1)
            z_edge = z_group[gather_idx]
        return z_edge, z_group, kl_loss

    def forward(
        self,
        data: Data,
        return_aux: bool = False,
        sample_latent: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.encode(data)
        h_i = x[data.pair_i] # one of the node in the predicted pair, shape [E, H]
        h_j = x[data.pair_j] # the other node in the predicted pair, shape [E, H]
        edge_group = self._edge_group_ids(data, num_edges=h_i.size(0), device=h_i.device)
        node_group = self._node_group_ids(data, x.size(0), h_i.device)
        num_groups = self._num_groups(
            node_group=node_group,
            edge_group=edge_group,
            group_attr_len=int(getattr(data, "branch_id_by_group").numel()) if hasattr(data, "branch_id_by_group") else 0,
        )
        pair_feat = torch.cat([h_i, h_j], dim=-1) # concatenated node features, shape [E, 2H]
        kl_loss = torch.zeros((), dtype=torch.float32, device=h_i.device)
        z_group = None
        # Optional molecule-balanced pooled feature by edge group.
        if self.cfg.molecule_balanced_pool:
            # Find all atoms within the group that each edge belongs to, compute the mean across each molecule, and then sum them up to obtain a vector representing the entire reactant environment.
            g_edge = self._edge_group_pooled_feature(x=x, data=data, edge_group=edge_group) # 找到每条边所属 group 里的所有原子，按分子做均值后求和，得到一个代表整个反应物环境的向量
            pair_feat = torch.cat([pair_feat, g_edge], dim=-1) #concatenate with the pair feature, shape [E, 2H+H]=[E, 3H]

        pair_feat = torch.cat([pair_feat, data.react_edge.view(-1, 1)], dim=-1)# add the react_edge feature, shape [E, 3H+1]

        if self.cfg.use_branch_feature:
            if self.cfg.branch_feature_mode == "scalar":
                b = self._edge_group_scalar_feature(
                    data=data,
                    edge_group=edge_group,
                    group_field_name="branch_value_by_group",
                    legacy_field_name="branch_value",
                    device=h_i.device,
                )
            else:
                b = self._contextual_branch_feature(x=x, data=data, edge_group=edge_group, device=h_i.device)
            pair_feat = torch.cat([pair_feat, b], dim=-1)

        if self.cfg.use_third_body_feature:
            t = self._edge_group_scalar_feature(
                data=data,
                edge_group=edge_group,
                group_field_name="third_body_by_group",
                legacy_field_name="third_body_value",
                device=h_i.device,
            )
            pair_feat = torch.cat([pair_feat, t], dim=-1)

        if self.cfg.use_latent_branching:
            z_edge, z_group, kl_loss = self._latent_edge_feature(
                x=x,
                data=data,
                edge_group=edge_group,
                sample_latent=sample_latent,
            )
            pair_feat = torch.cat([pair_feat, z_edge], dim=-1)

        logits = self.edge_decoder(pair_feat, edge_group=edge_group)
        rate_pred_by_group = None
        if self.cfg.predict_rate_coeffs:
            rate_feat = self._pooled_group_context(
                x=x,
                data=data,
                node_group=node_group,
                num_groups=num_groups,
            )

            if self.cfg.use_branch_feature:
                if self.cfg.branch_feature_mode == "scalar":
                    b_group = self._group_scalar_feature(
                        data=data,
                        num_groups=num_groups,
                        group_field_name="branch_value_by_group",
                        legacy_field_name="branch_value",
                        device=h_i.device,
                    )
                else:
                    b_group = self._contextual_branch_feature_by_group(
                        x=x,
                        data=data,
                        node_group=node_group,
                        num_groups=num_groups,
                        device=h_i.device,
                    )
                rate_feat = torch.cat([rate_feat, b_group], dim=-1)

            if self.cfg.use_third_body_feature:
                t_group = self._group_scalar_feature(
                    data=data,
                    num_groups=num_groups,
                    group_field_name="third_body_by_group",
                    legacy_field_name="third_body_value",
                    device=h_i.device,
                )
                rate_feat = torch.cat([rate_feat, t_group], dim=-1)

            if self.cfg.use_latent_branching:
                if z_group is None:
                    z_group = torch.zeros((num_groups, self.cfg.latent_dim), dtype=torch.float32, device=h_i.device)
                rate_feat = torch.cat([rate_feat, z_group], dim=-1)

            rate_pred_by_group = self.rate_mlp(rate_feat)

        if return_aux:
            aux = {"kl_loss": kl_loss}
            if rate_pred_by_group is not None:
                aux["rate_pred_by_group"] = rate_pred_by_group
            return logits, aux
        return logits
