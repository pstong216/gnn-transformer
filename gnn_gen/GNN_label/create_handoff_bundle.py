#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_KEY_FILES = [
    "README.md",
    "hydrogen_adjacency.py",
    "data_prep.py",
    "model.py",
    "train.py",
    "validate.py",
    "plotting.py",
    "reaction_dataset_prediction.py",
    "reaction_stochastic_inference.py",
    "reaction_inference_copies.py",
    "refresh_demo.py",
    "plot_model_diagram.py",
]

CONFIG_FILES = [
    "reaction_dataset_prediction.py",
    "reaction_stochastic_inference.py",
]


def _run(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as exc:  # defensive
        return 1, "", str(exc)


def _git_info(repo: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"is_git_repo": False}
    rc, out, _ = _run(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo)
    if rc != 0 or out.strip() != "true":
        return info

    info["is_git_repo"] = True
    _, branch, _ = _run(["git", "branch", "--show-current"], cwd=repo)
    _, commit, _ = _run(["git", "rev-parse", "HEAD"], cwd=repo)
    _, status, _ = _run(["git", "status", "--short", "--branch"], cwd=repo)
    _, remotes, _ = _run(["git", "remote", "-v"], cwd=repo)
    _, log, _ = _run(["git", "--no-pager", "log", "--oneline", "-n", "12"], cwd=repo)

    info.update(
        {
            "branch": branch,
            "commit": commit,
            "status_short_branch": status.splitlines(),
            "remotes": remotes.splitlines(),
            "recent_commits": log.splitlines(),
        }
    )
    return info


def _python_ml_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
    }
    try:
        import torch  # type: ignore

        info["torch_version"] = getattr(torch, "__version__", "unknown")
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            info["cuda_device_name_0"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = getattr(torch.version, "cuda", None)
    except Exception as exc:
        info["torch_error"] = str(exc)

    try:
        import torch_geometric  # type: ignore

        info["torch_geometric_version"] = getattr(torch_geometric, "__version__", "unknown")
    except Exception as exc:
        info["torch_geometric_error"] = str(exc)

    return info


def _system_info() -> Dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "user": os.environ.get("USER", ""),
        "cwd": str(Path.cwd()),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "venv": os.environ.get("VIRTUAL_ENV"),
    }


def _literal_eval_safe(node: ast.AST) -> Optional[Any]:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _extract_uppercase_constants(path: Path) -> Dict[str, Any]:
    constants: Dict[str, Any] = {}
    if not path.exists():
        return constants
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except Exception:
        return constants

    for node in tree.body:
        if isinstance(node, ast.Assign):
            value = _literal_eval_safe(node.value)
            if value is None:
                continue
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id.isupper():
                    constants[tgt.id] = value
        elif isinstance(node, ast.AnnAssign):
            if not isinstance(node.target, ast.Name):
                continue
            if not node.target.id.isupper():
                continue
            if node.value is None:
                continue
            value = _literal_eval_safe(node.value)
            if value is None:
                continue
            constants[node.target.id] = value
    return constants


def _find_absolute_paths(path: Path) -> List[str]:
    out: List[str] = []
    if not path.exists():
        return out
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return out
    for token in ['"', "'"]:
        parts = text.split(token)
        for i in range(1, len(parts), 2):
            s = parts[i]
            if s.startswith("/") and len(s) > 1:
                out.append(s)
    seen = set()
    dedup = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        dedup.append(p)
    return dedup


def _find_recent_artifacts(repo: Path, limit: int = 20) -> List[Dict[str, Any]]:
    patterns = [
        "**/model.pt",
        "**/losses.pt",
        "**/snapshots.pt",
        "**/summary.txt",
        "**/events.txt",
    ]
    found: List[Tuple[float, Path]] = []
    for pat in patterns:
        for p in repo.glob(pat):
            try:
                if p.is_file():
                    found.append((p.stat().st_mtime, p))
            except FileNotFoundError:
                continue
    found.sort(key=lambda x: x[0], reverse=True)

    rows: List[Dict[str, Any]] = []
    for ts, p in found[:limit]:
        rows.append(
            {
                "path": str(p.relative_to(repo)),
                "mtime_utc": datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z"),
                "size_bytes": p.stat().st_size,
            }
        )
    return rows


def _copy_key_files(repo: Path, out_dir: Path, key_files: List[str]) -> List[str]:
    copied: List[str] = []
    files_dir = out_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    for rel in key_files:
        src = repo / rel
        if not src.exists() or not src.is_file():
            continue
        dst = files_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)
    return copied


def _copy_optional_codex_config(out_dir: Path) -> List[str]:
    copied: List[str] = []
    codex_home = Path.home() / ".codex"
    if not codex_home.exists():
        return copied
    dst_root = out_dir / "codex_home"
    dst_root.mkdir(parents=True, exist_ok=True)

    candidates = [
        codex_home / "config.toml",
        codex_home / "AGENTS.md",
    ]
    for src in candidates:
        if src.exists() and src.is_file():
            dst = dst_root / src.name
            shutil.copy2(src, dst)
            copied.append(str(src))
    return copied


def _write_handoff_md(
    out_path: Path,
    repo: Path,
    metadata: Dict[str, Any],
    copied_files: List[str],
    copied_codex: List[str],
) -> None:
    git = metadata.get("git", {})
    sys_info = metadata.get("system", {})
    py_info = metadata.get("python_ml", {})
    configs = metadata.get("config_constants", {})
    abs_paths = metadata.get("absolute_paths", {})
    artifacts = metadata.get("recent_artifacts", [])

    lines: List[str] = []
    lines.append("# Handoff Bundle")
    lines.append("")
    lines.append(f"Generated at: `{sys_info.get('timestamp_utc', '')}`")
    lines.append(f"Repo root: `{repo}`")
    lines.append("")
    lines.append("## Quick Resume On New Server")
    lines.append("")
    lines.append("1. Clone and enter repo:")
    lines.append("   `git clone <repo-url> && cd <repo-name>`")
    lines.append("2. Checkout exact commit from this bundle:")
    lines.append(f"   `git checkout {git.get('commit', '<commit>')}`")
    lines.append("3. Recreate environment (same Python/torch stack).")
    lines.append("4. Update any absolute paths listed below.")
    lines.append("5. Start from entrypoint:")
    lines.append("   `python reaction_dataset_prediction.py`")
    lines.append("")

    lines.append("## Git State")
    lines.append("")
    lines.append(f"- Branch: `{git.get('branch', '')}`")
    lines.append(f"- Commit: `{git.get('commit', '')}`")
    lines.append("")
    lines.append("### Remotes")
    lines.append("")
    for r in git.get("remotes", []):
        lines.append(f"- `{r}`")
    lines.append("")
    lines.append("### Status")
    lines.append("")
    for s in git.get("status_short_branch", []):
        lines.append(f"- `{s}`")
    lines.append("")
    lines.append("### Recent Commits")
    lines.append("")
    for c in git.get("recent_commits", []):
        lines.append(f"- `{c}`")
    lines.append("")

    lines.append("## Runtime Environment")
    lines.append("")
    lines.append(f"- Host: `{sys_info.get('hostname', '')}`")
    lines.append(f"- Platform: `{sys_info.get('platform', '')}`")
    lines.append(f"- User: `{sys_info.get('user', '')}`")
    lines.append(f"- Conda env: `{sys_info.get('conda_env', '')}`")
    lines.append(f"- Virtual env: `{sys_info.get('venv', '')}`")
    lines.append(f"- Python: `{py_info.get('python_version', '')}`")
    lines.append(f"- Python exe: `{py_info.get('python_executable', '')}`")
    lines.append(f"- Torch: `{py_info.get('torch_version', py_info.get('torch_error', 'n/a'))}`")
    lines.append(f"- CUDA available: `{py_info.get('cuda_available', 'n/a')}`")
    lines.append(f"- PyG: `{py_info.get('torch_geometric_version', py_info.get('torch_geometric_error', 'n/a'))}`")
    lines.append("")

    lines.append("## Key Config Constants")
    lines.append("")
    for rel, kv in configs.items():
        lines.append(f"### `{rel}`")
        lines.append("")
        for k in sorted(kv.keys()):
            lines.append(f"- `{k}`: `{kv[k]}`")
        lines.append("")

    lines.append("## Absolute Paths To Review")
    lines.append("")
    for rel, paths in abs_paths.items():
        lines.append(f"### `{rel}`")
        lines.append("")
        if not paths:
            lines.append("- None found")
        else:
            for p in paths:
                lines.append(f"- `{p}`")
        lines.append("")

    lines.append("## Copied Files")
    lines.append("")
    for rel in copied_files:
        lines.append(f"- `{rel}`")
    lines.append("")

    if copied_codex:
        lines.append("## Copied Codex Config")
        lines.append("")
        for src in copied_codex:
            lines.append(f"- `{src}`")
        lines.append("")

    lines.append("## Recent Artifacts")
    lines.append("")
    for a in artifacts:
        lines.append(f"- `{a['path']}` | `{a['mtime_utc']}` | `{a['size_bytes']} bytes`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This bundle captures project/runtime state; local IDE thread history is not exported.")
    lines.append("- For Codex IDE continuity across devices, use cloud task follow-up or paste `handoff_prompt.md` into a new chat.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_handoff_prompt(out_path: Path, metadata: Dict[str, Any]) -> None:
    git = metadata.get("git", {})
    prompt = f"""Continue this project from a prior server handoff.

Repository state:
- Branch: {git.get("branch", "")}
- Commit: {git.get("commit", "")}

Primary objective:
- Resume work in `GNN_label` with current training/inference pipeline.

Please start by:
1. Reading `README.md`.
2. Reading `reaction_dataset_prediction.py`, `data_prep.py`, `model.py`, `train.py`, `validate.py`, and `reaction_stochastic_inference.py`.
3. Summarizing current toggles and the fastest way to reproduce the last run.
4. Identifying any absolute paths that need patching for this server.

Constraints:
- Keep existing behavior unless explicitly asked to change it.
- Preserve output directory conventions unless they are path-broken here.
"""
    out_path.write_text(prompt, encoding="utf-8")


def _make_archive(src_dir: Path, archive_path: Path) -> None:
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(src_dir, arcname=src_dir.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a portable Codex handoff bundle for this repo.")
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repository root (default: script directory).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <repo>/handoff_bundle_<timestamp>).",
    )
    parser.add_argument(
        "--include-codex-config",
        action="store_true",
        help="Copy ~/.codex/config.toml and ~/.codex/AGENTS.md into the bundle.",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Do not create a .tar.gz archive.",
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out_dir = args.out.resolve() if args.out else (repo / f"handoff_bundle_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata: Dict[str, Any] = {}
    metadata["system"] = _system_info()
    metadata["git"] = _git_info(repo)
    metadata["python_ml"] = _python_ml_info()
    metadata["recent_artifacts"] = _find_recent_artifacts(repo, limit=25)

    config_constants: Dict[str, Dict[str, Any]] = {}
    absolute_paths: Dict[str, List[str]] = {}
    for rel in CONFIG_FILES:
        p = repo / rel
        config_constants[rel] = _extract_uppercase_constants(p)
        absolute_paths[rel] = _find_absolute_paths(p)
    metadata["config_constants"] = config_constants
    metadata["absolute_paths"] = absolute_paths

    copied_files = _copy_key_files(repo, out_dir, DEFAULT_KEY_FILES)
    copied_codex: List[str] = []
    if args.include_codex_config:
        copied_codex = _copy_optional_codex_config(out_dir)

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    _write_handoff_md(out_dir / "HANDOFF.md", repo, metadata, copied_files, copied_codex)
    _write_handoff_prompt(out_dir / "handoff_prompt.md", metadata)

    archive_path = out_dir.with_suffix(".tar.gz")
    if not args.no_archive:
        _make_archive(out_dir, archive_path)

    print(f"[handoff] bundle directory: {out_dir}")
    print(f"[handoff] metadata: {out_dir / 'metadata.json'}")
    print(f"[handoff] summary: {out_dir / 'HANDOFF.md'}")
    print(f"[handoff] prompt: {out_dir / 'handoff_prompt.md'}")
    print(f"[handoff] copied key files: {len(copied_files)}")
    if args.include_codex_config:
        print(f"[handoff] copied codex config files: {len(copied_codex)}")
    if not args.no_archive:
        print(f"[handoff] archive: {archive_path}")


if __name__ == "__main__":
    main()
