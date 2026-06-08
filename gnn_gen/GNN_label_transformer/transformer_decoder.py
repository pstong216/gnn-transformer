from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class EdgeDecoderConfig:
    input_dim: int
    num_classes: int
    decoder_type: str = "mlp"  # "mlp" or "transformer"
    edge_hidden: int = 64
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_dropout: float = 0.1
    max_groups: int = 32


class MLPDecoder(nn.Module):
    def __init__(self, cfg: EdgeDecoderConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.edge_hidden),
            nn.ReLU(),
            nn.Linear(cfg.edge_hidden, cfg.num_classes),
        )

    def forward(self, pair_feat: torch.Tensor, edge_group: torch.Tensor | None = None) -> torch.Tensor:
        del edge_group
        return self.net(pair_feat)


class TransformerEdgeDecoder(nn.Module):
    def __init__(self, cfg: EdgeDecoderConfig):
        super().__init__()
        if cfg.input_dim % cfg.transformer_heads != 0:
            raise ValueError(
                f"decoder input_dim ({cfg.input_dim}) must be divisible by transformer_heads ({cfg.transformer_heads})"
            )
        self.cfg = cfg
        self.group_embed = nn.Embedding(cfg.max_groups, cfg.input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.input_dim,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.input_dim * 4,
            dropout=cfg.transformer_dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.transformer_layers)
        self.out_proj = nn.Linear(cfg.input_dim, cfg.num_classes)

    def _group_attention_mask(self, edge_group: torch.Tensor) -> torch.Tensor:
        # True entries are blocked attention positions.
        if edge_group.numel() == 0:
            return torch.zeros((0, 0), dtype=torch.bool, device=edge_group.device)
        return edge_group.view(-1, 1) != edge_group.view(1, -1)

    def forward(self, pair_feat: torch.Tensor, edge_group: torch.Tensor | None = None) -> torch.Tensor:
        if pair_feat.numel() == 0:
            return self.out_proj(pair_feat)

        if edge_group is None or edge_group.numel() != pair_feat.size(0):
            edge_group = torch.zeros(pair_feat.size(0), dtype=torch.long, device=pair_feat.device)
        else:
            edge_group = edge_group.long().to(pair_feat.device)

        group_idx = torch.clamp(edge_group, min=0, max=self.cfg.max_groups - 1)
        seq = pair_feat + self.group_embed(group_idx)
        attn_mask = self._group_attention_mask(group_idx)
        seq = self.encoder(seq.unsqueeze(0), mask=attn_mask).squeeze(0)
        return self.out_proj(seq)


def build_edge_decoder(cfg: EdgeDecoderConfig) -> nn.Module:
    if cfg.decoder_type == "mlp":
        return MLPDecoder(cfg)
    if cfg.decoder_type == "transformer":
        return TransformerEdgeDecoder(cfg)
    raise ValueError(f"Unknown decoder_type={cfg.decoder_type!r}. Expected 'mlp' or 'transformer'.")
