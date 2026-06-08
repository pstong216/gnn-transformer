from typing import Dict, List, Tuple
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv

import networkx as nx

from hydrogen_reactions import SPECIES as HYDROGEN_SPECIES
from FFCM2_reactions import SPECIES as CARBON_SPECIES


NUM_CLASSES = 4  # bond order 0..3
INDEX_SCALE = 1.0

EPOCHS = 500
LR = 1e-3
HIDDEN = 64
LAYERS = 3
EDGE_HIDDEN = 64

OUT_ROOT = "species_tokenizer_runs"

ALL_SPECIES = {**CARBON_SPECIES, **HYDROGEN_SPECIES}
ALL_TYPES = sorted({atom for mol in ALL_SPECIES.values() for atom in mol.atom_types})
TYPE_INDEX = {t: i for i, t in enumerate(ALL_TYPES)}
TRAIN_NAMES = [name for name, mol in ALL_SPECIES.items() if len(mol.atom_types) > 1]
SKIPPED_NAMES = [name for name, mol in ALL_SPECIES.items() if len(mol.atom_types) <= 1]


def _type_index_features(atom_types: List[str]) -> torch.Tensor:
    type_counts: Dict[str, int] = {}
    per_type_idx: List[int] = []
    for a in atom_types:
        type_counts.setdefault(a, 0)
        per_type_idx.append(type_counts[a])
        type_counts[a] += 1
    type_totals = {t: max(1, c - 1) for t, c in type_counts.items()}
    rows: List[List[float]] = []
    for i, a in enumerate(atom_types):
        norm_idx = (per_type_idx[i] / type_totals[a]) * INDEX_SCALE
        one_hot = [0.0] * len(ALL_TYPES)
        idx_hot = [0.0] * len(ALL_TYPES)
        pos = TYPE_INDEX[a]
        one_hot[pos] = 1.0
        idx_hot[pos] = norm_idx
        rows.append(one_hot + idx_hot)
    return torch.tensor(rows, dtype=torch.float32)


def build_molecule_data(name: str) -> Tuple[Data, torch.Tensor, List[str]]:
    mol = ALL_SPECIES[name]
    atom_types = mol.atom_types
    adj = torch.tensor(mol.adjacency, dtype=torch.float32)
    n = len(atom_types)

    pair_index = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_i = torch.tensor([i for i, _ in pair_index], dtype=torch.long)
    pair_j = torch.tensor([j for _, j in pair_index], dtype=torch.long)

    x = _type_index_features(atom_types)

    edges = []
    edge_attrs = []
    for i in range(n):
        for j in range(i + 1, n):
            bond = mol.adjacency[i][j]
            if bond != 0:
                edges.append([i, j])
                edge_attrs.append([bond])
                edges.append([j, i])
                edge_attrs.append([bond])
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)

    y_edge = adj[pair_i, pair_j].long().clamp(0, NUM_CLASSES - 1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.pair_i = pair_i
    data.pair_j = pair_j
    data.y_edge = y_edge
    data.adj = adj
    data.atom_types = atom_types
    data.name = name
    return data, adj, atom_types


class MoleculeAutoEncoder(nn.Module):
    def __init__(self, node_in: int, hidden: int = HIDDEN, layers: int = LAYERS, edge_hidden: int = EDGE_HIDDEN):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(layers):
            mlp = nn.Sequential(
                nn.Linear(node_in if i == 0 else hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(mlp, edge_dim=1, train_eps=False))
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 2, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, NUM_CLASSES),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
        h_i = x[data.pair_i]
        h_j = x[data.pair_j]
        pair_feat = torch.cat([h_i, h_j], dim=-1)
        logits = self.edge_mlp(pair_feat)
        return logits

    def encode(self, data: Data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
        return x


def plot_graph(adj_mat: torch.Tensor, atom_types: List[str], fname: str, title: str) -> None:
    G = nx.Graph()
    n = len(atom_types)
    for i, a in enumerate(atom_types):
        G.add_node(i, atom=a)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(adj_mat[i, j].item())
            if w != 0:
                G.add_edge(i, j, weight=w)
    pos = nx.spring_layout(G, seed=42)
    color_map = {
        "H": "#1976d2",
        "O": "#d32f2f",
        "C": "#424242",
        "N": "#388e3c",
        "S": "#f9a825",
    }
    node_colors = [color_map.get(atom_types[i], "#7b7b7b") for i in G.nodes()]
    labels = {i: f"{atom_types[i]}_{i}" for i in G.nodes()}
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    single_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] == 1]
    double_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] == 2]
    triple_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] == 3]
    nx.draw_networkx_edges(G, pos, edgelist=single_edges, width=1.2, edge_color="#555")
    if double_edges:
        nx.draw_networkx_edges(G, pos, edgelist=double_edges, width=2.5, edge_color="#555")
    if triple_edges:
        nx.draw_networkx_edges(G, pos, edgelist=triple_edges, width=3.5, edge_color="#555")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


def train_all() -> None:
    data_list = []
    names = list(ALL_SPECIES.keys())
    total_species = len(names)
    if SKIPPED_NAMES:
        print(f"Skipping single-atom species in training: {', '.join(SKIPPED_NAMES)}")
    kept = 0
    for idx, name in enumerate(names, start=1):
        data, _, atom_types = build_molecule_data(name)
        if len(atom_types) < 2:
            continue
        data_list.append(data)
        kept += 1
        if idx % 10 == 0 or idx == total_species:
            print(f"Built dataset {idx}/{total_species} species (kept {kept})")

    # global class weights
    all_edges = torch.cat([d.y_edge for d in data_list], dim=0)
    counts = torch.bincount(all_edges, minlength=NUM_CLASSES).float()
    weights = counts.sum() / (NUM_CLASSES * (counts + 1e-6))
    weights = weights / weights.mean()

    loader = DataLoader(data_list, batch_size=1, shuffle=True)
    node_in = data_list[0].x.size(1)
    model = MoleculeAutoEncoder(node_in=node_in)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.y_edge.view(-1), weight=weights.to(logits.device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
        if epoch == 1 or epoch % 25 == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch}/{EPOCHS} - loss {total_loss:.4f}")

    os.makedirs(OUT_ROOT, exist_ok=True)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, EPOCHS + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Molecule Autoencoder Training Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_ROOT, "loss.png"), dpi=200)
    plt.close()

    model.eval()
    for name in TRAIN_NAMES:
        data, true_adj, atom_types = build_molecule_data(name)
        with torch.no_grad():
            embeddings = model.encode(data)
            logits = model(data)
            pred = logits.argmax(dim=-1)
        n = len(atom_types)
        pred_mat = torch.zeros((n, n), dtype=torch.float32)
        for idx, (i, j) in enumerate(zip(data.pair_i.tolist(), data.pair_j.tolist())):
            bond = float(pred[idx].item())
            pred_mat[i, j] = bond
            pred_mat[j, i] = bond
        out_dir = os.path.join(OUT_ROOT, name)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "pred_mat.txt"), "w") as f:
            for row in pred_mat.tolist():
                f.write(" ".join(f"{v:.1f}" for v in row) + "\n")
        with open(os.path.join(out_dir, "true_mat.txt"), "w") as f:
            for row in true_adj.tolist():
                f.write(" ".join(f"{v:.1f}" for v in row) + "\n")
        with open(os.path.join(out_dir, "node_embeddings.txt"), "w") as f:
            f.write("node\tatom\t" + "\t".join(f"dim{i}" for i in range(embeddings.size(1))) + "\n")
            for idx, atom in enumerate(atom_types):
                vals = "\t".join(f"{v:.6f}" for v in embeddings[idx].tolist())
                f.write(f"{idx}\t{atom}\t{vals}\n")
        plot_graph(true_adj, atom_types, os.path.join(out_dir, "graph_true.png"), f"{name} True")
        plot_graph(pred_mat, atom_types, os.path.join(out_dir, "graph_pred.png"), f"{name} Pred")
        err = float((pred_mat - true_adj).abs().sum().item())
        with open(os.path.join(out_dir, "error.txt"), "w") as f:
            f.write(f"abs_error {err:.1f}\n")
        print(f"{name}: abs_error {err:.1f}")


if __name__ == "__main__":
    train_all()
