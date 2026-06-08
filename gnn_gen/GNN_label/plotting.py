from __future__ import annotations

from typing import List
import os

import matplotlib.pyplot as plt
import networkx as nx
import torch


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def plot_loss(losses: List[float], out_path: str) -> None:
    _ensure_parent_dir(out_path)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_graph(adj_mat: torch.Tensor, atom_types: List[str], out_path: str, title: str) -> None:
    _ensure_parent_dir(out_path)
    graph = nx.Graph()
    n = len(atom_types)
    for i, atom in enumerate(atom_types):
        graph.add_node(i, atom=atom)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(adj_mat[i, j].item())
            if w != 0:
                graph.add_edge(i, j, weight=w)

    pos = nx.spring_layout(graph, seed=42)
    labels = {i: f"{atom_types[i]}_{i}" for i in graph.nodes()}
    colors = ["#d32f2f" if atom_types[i] == "O" else "#1976d2" for i in graph.nodes()]

    single = [(u, v) for u, v, d in graph.edges(data=True) if d["weight"] == 1]
    double = [(u, v) for u, v, d in graph.edges(data=True) if d["weight"] == 2]
    triple = [(u, v) for u, v, d in graph.edges(data=True) if d["weight"] == 3]

    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(graph, pos, node_color=colors)
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)
    nx.draw_networkx_edges(graph, pos, edgelist=single, width=1.3, edge_color="#555")
    if double:
        nx.draw_networkx_edges(graph, pos, edgelist=double, width=2.7, edge_color="#555")
    if triple:
        nx.draw_networkx_edges(graph, pos, edgelist=triple, width=3.7, edge_color="#555")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
