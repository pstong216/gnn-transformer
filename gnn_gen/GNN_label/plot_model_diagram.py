from __future__ import annotations

import argparse
import inspect
import os
from typing import Any, Dict, Optional

import torch

from data_prep import DatasetConfig, build_datasets, num_classes_and_offset
from model import EdgePredictor, ModelConfig


def _pick_predict_mode(
    cli_value: Optional[bool],
    checkpoint: Optional[Dict[str, Any]],
) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    if checkpoint is not None and "predict_bond_change" in checkpoint:
        return bool(checkpoint["predict_bond_change"])
    return True


def _build_minimal_sample(predict_bond_change: bool) -> Any:
    cfg = DatasetConfig(
        predict_bond_change=predict_bond_change,
        selected_reactions=[1],
        filter_unique_reactants=True,
        perturb_training=False,
        include_base_sample=True,
        copy_counts=[1],
        random_group_training=False,
        random_group_eval=False,
        show_progress=False,
    )
    bundle = build_datasets(cfg)
    if not bundle.train_data:
        raise RuntimeError("Failed to build a sample graph for diagram generation.")
    return bundle.train_data[0]


def _model_cfg_from_checkpoint_or_defaults(
    sample: Any,
    predict_bond_change: bool,
    checkpoint_cfg: Optional[Dict[str, Any]],
) -> ModelConfig:
    num_classes, _ = num_classes_and_offset(predict_bond_change)
    sig = inspect.signature(ModelConfig)
    kwargs: Dict[str, Any] = {}

    for name, param in sig.parameters.items():
        if checkpoint_cfg is not None and name in checkpoint_cfg:
            kwargs[name] = checkpoint_cfg[name]
            continue
        if name == "node_in":
            kwargs[name] = int(sample.x.size(1))
            continue
        if name == "num_classes":
            kwargs[name] = int(num_classes)
            continue
        if param.default is not inspect._empty:
            kwargs[name] = param.default
            continue
        raise RuntimeError(
            f"Cannot infer ModelConfig field '{name}'. "
            "Provide a checkpoint with model_cfg or update this script."
        )
    return ModelConfig(**kwargs)


def _load_checkpoint(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu")


def _load_weights_if_available(model: EdgePredictor, checkpoint: Optional[Dict[str, Any]]) -> None:
    if checkpoint is None:
        return
    state = checkpoint.get("state_dict")
    if state is None:
        state = checkpoint.get("model_state")
    if state is None:
        return

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[weights] missing keys: {len(missing)}")
    if unexpected:
        print(f"[weights] unexpected keys: {len(unexpected)}")


def _extract_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            t = _extract_tensor(item)
            if t is not None:
                return t
    if isinstance(output, dict):
        for _, item in output.items():
            t = _extract_tensor(item)
            if t is not None:
                return t
    raise RuntimeError("Could not find a tensor output from model forward pass.")


def _save_model_summary(model: EdgePredictor, out_path: str) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Model class: EdgePredictor\n")
        f.write(f"Total parameters: {total}\n")
        f.write(f"Trainable parameters: {trainable}\n\n")
        f.write("Module tree:\n")
        f.write(str(model))
        f.write("\n")


def _draw_with_torchview(model: EdgePredictor, sample: Any, out_base: str, fmt: str, device: str, depth: int) -> str:
    from torchview import draw_graph  # type: ignore

    graph = draw_graph(
        model,
        input_data=(sample,),
        expand_nested=True,
        depth=depth,
        device=device,
        graph_name="EdgePredictor",
    )
    graph.visual_graph.render(out_base, format=fmt, cleanup=True)
    return f"{out_base}.{fmt}"


def _draw_with_torchviz(model: EdgePredictor, sample: Any, out_base: str, fmt: str, device: str) -> str:
    from torchviz import make_dot  # type: ignore

    model.eval()
    # Keep autograd enabled so torchviz can trace the full computation graph.
    out = model(sample)
    out_t = _extract_tensor(out)
    dot = make_dot(out_t, params=dict(model.named_parameters()), show_attrs=False, show_saved=False)
    dot.graph_attr.update(rankdir="LR", dpi="220")
    dot.render(out_base, format=fmt, cleanup=True)
    return f"{out_base}.{fmt}"


def _linear_layers_from_sequential(seq: Any) -> str:
    parts = []
    if hasattr(seq, "__iter__"):
        for m in seq:
            if isinstance(m, torch.nn.Linear):
                parts.append(f"Linear({m.in_features}->{m.out_features})")
            else:
                parts.append(m.__class__.__name__)
    return "\\n".join(parts) if parts else str(seq.__class__.__name__)


def _draw_architecture_diagram(model: EdgePredictor, out_base: str, fmt: str) -> str:
    from graphviz import Digraph  # type: ignore

    cfg = model.cfg
    dot = Digraph("EdgePredictorArchitecture")
    dot.attr(rankdir="LR", dpi="220", splines="spline", nodesep="0.45", ranksep="0.55")
    dot.attr("node", shape="box", style="rounded,filled", fillcolor="#f6f8fa", color="#3b3b3b", fontsize="10")

    enc_lines = [f"GINE stack: {len(model.convs)} layers", f"node_in={cfg.node_in}", f"hidden={cfg.hidden}"]
    dot.node("in", "Input Data\\n(x, edge_index, edge_attr, pair_i/pair_j)")
    dot.node("enc", "\\n".join(enc_lines))
    dot.node("hij", "Pair Features\\n[h_i || h_j]")

    dot.edge("in", "enc", label="message passing")
    dot.edge("enc", "hij", label="gather pair nodes")

    if cfg.molecule_balanced_pool:
        dot.node("pool", "Group-aware Pool\\n(node_group_id + pair_group_id)\\nmean per molecule, then sum")
        dot.edge("enc", "pool")
        dot.edge("pool", "cat", label="+ pooled context")

    dot.node("react", "React bond feature\\nreact_edge")
    dot.edge("react", "cat")
    dot.edge("hij", "cat")

    if cfg.use_branch_feature:
        if cfg.branch_feature_mode == "scalar":
            dot.node("branch", "Branch scalar\\nbranch_value_by_group")
        else:
            dot.node(
                "branch",
                "Contextual branch\\nEmbedding(branch_id_by_group) + pooled context\\nbranch_fuse MLP",
            )
        dot.edge("branch", "cat")

    if cfg.use_third_body_feature:
        dot.node("third", "Third-body scalar\\nthird_body_by_group")
        dot.edge("third", "cat")

    dot.node("cat", "Concatenate edge decoder inputs")
    dot.node("edge_mlp", f"Edge MLP\\n{_linear_layers_from_sequential(model.edge_mlp)}")
    dot.node("out", f"Output logits\\nnum_classes={cfg.num_classes}")
    dot.edge("cat", "edge_mlp")
    dot.edge("edge_mlp", "out")

    dot.render(out_base, format=fmt, cleanup=True)
    return f"{out_base}.{fmt}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate NN architecture diagrams for GNN_label EdgePredictor."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/users/3/du000298/GNN_label/reaction_dataset_prediction/model.pt",
        help="Checkpoint path. If missing, random model weights are used.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/users/3/du000298/GNN_label/model_diagram",
        help="Output directory.",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default="edge_predictor",
        help="Base filename for outputs.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format for graphviz render.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "torchview", "torchviz", "architecture"],
        help="Diagram backend. auto tries torchview -> torchviz -> architecture.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu or cuda. cpu is safer for rendering tools.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=6,
        help="torchview module expansion depth.",
    )
    parser.add_argument(
        "--predict-bond-change",
        type=int,
        default=None,
        choices=[0, 1],
        help="Override target mode. 1=bond-change, 0=direct bond.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = _load_checkpoint(args.checkpoint)
    predict_bond_change = _pick_predict_mode(
        cli_value=None if args.predict_bond_change is None else bool(args.predict_bond_change),
        checkpoint=ckpt,
    )
    sample = _build_minimal_sample(predict_bond_change=predict_bond_change)

    checkpoint_cfg = None
    if ckpt is not None and isinstance(ckpt.get("model_cfg"), dict):
        checkpoint_cfg = ckpt["model_cfg"]

    model_cfg = _model_cfg_from_checkpoint_or_defaults(
        sample=sample,
        predict_bond_change=predict_bond_change,
        checkpoint_cfg=checkpoint_cfg,
    )
    model = EdgePredictor(model_cfg)
    _load_weights_if_available(model, ckpt)
    model.to(args.device).eval()
    sample = sample.to(args.device)

    summary_path = os.path.join(args.out_dir, f"{args.basename}_summary.txt")
    _save_model_summary(model, summary_path)
    print(f"[ok] wrote model summary: {summary_path}")

    out_base = os.path.join(args.out_dir, args.basename)

    backend_order = [args.backend]
    if args.backend == "auto":
        backend_order = ["torchview", "torchviz", "architecture"]

    errors = []
    for backend in backend_order:
        try:
            if backend == "torchview":
                out_file = _draw_with_torchview(
                    model=model,
                    sample=sample,
                    out_base=f"{out_base}_torchview",
                    fmt=args.format,
                    device=args.device,
                    depth=max(1, int(args.depth)),
                )
            elif backend == "torchviz":
                out_file = _draw_with_torchviz(
                    model=model,
                    sample=sample,
                    out_base=f"{out_base}_torchviz",
                    fmt=args.format,
                    device=args.device,
                )
            elif backend == "architecture":
                out_file = _draw_architecture_diagram(
                    model=model,
                    out_base=f"{out_base}_architecture",
                    fmt=args.format,
                )
            else:
                raise ValueError(f"Unsupported backend: {backend}")
            print(f"[ok] wrote diagram ({backend}): {out_file}")
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{backend}: {exc}")
            print(f"[warn] backend failed: {backend} -> {exc}")
    else:
        msg = "\n".join(errors)
        raise RuntimeError(
            "All diagram backends failed.\n"
            "Install one of: pip install torchview graphviz OR pip install torchviz.\n"
            f"Details:\n{msg}"
        )


if __name__ == "__main__":
    main()
