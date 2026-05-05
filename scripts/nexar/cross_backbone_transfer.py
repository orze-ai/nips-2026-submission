#!/usr/bin/env python3
"""cross_backbone_transfer.py — Cross-backbone transfer experiment (Table 5).

Evaluates how champion hyperparameters transfer across different video
backbones.  Given a champion config YAML and a list of backbone model names,
retrains with the champion's HPs but swaps the backbone, then compares
test-set mAP.

Usage:
    python scripts/nexar/cross_backbone_transfer.py \
        --config configs/nexar/champion/vjepa2_alertonly_v4.yaml \
        --backbones facebook/vjepa2-vitl-fpc64-256 \
                    facebook/vjepa2-vitb-fpc64-256 \
                    MCG-NJU/videomae-huge \
                    microsoft/xclip-large-patch14 \
        --output table5_results.json

    # Or with pre-trained checkpoints (skip training, evaluate only):
    python scripts/nexar/cross_backbone_transfer.py \
        --vjepa /path/to/vjepa2_best.pt \
        --alt /path/to/alt_backbone_best.pt \
        --output xfer.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent

# Default backbones to compare (Table 5 in the paper)
DEFAULT_BACKBONES = [
    "facebook/vjepa2-vitl-fpc64-256",    # Champion backbone
    "facebook/vjepa2-vitb-fpc64-256",     # V-JEPA 2 ViT-B
    "MCG-NJU/videomae-huge",             # VideoMAE-Huge
    "microsoft/xclip-large-patch14",      # X-CLIP Large
]


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config file."""
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML required: pip install pyyaml")
        sys.exit(1)

    with open(config_path) as f:
        return yaml.safe_load(f)


def train_with_backbone(
    config: Dict[str, Any],
    backbone: str,
    output_dir: str,
    train_script: str,
) -> Optional[str]:
    """Train a model with a specific backbone, return best checkpoint path."""
    import tempfile
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML required: pip install pyyaml")
        return None

    # Modify config for this backbone
    run_config = dict(config)
    run_config["model_name"] = backbone
    run_config["output_dir"] = output_dir

    # Write temporary config
    tmp_config = os.path.join(output_dir, "config.yaml")
    os.makedirs(output_dir, exist_ok=True)
    with open(tmp_config, "w") as f:
        yaml.dump(run_config, f)

    cmd = [sys.executable, train_script, "--config", tmp_config]
    logger.info("Training backbone: %s", backbone)
    logger.info("  Command: %s", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error("Training failed for %s: %s", backbone, e)
        return None

    # Look for best checkpoint
    best_ckpt = os.path.join(output_dir, "best_model.pt")
    if os.path.isfile(best_ckpt):
        return best_ckpt

    # Fallback: look for any .pt file
    for f in sorted(Path(output_dir).glob("*.pt")):
        return str(f)

    logger.error("No checkpoint found after training %s", backbone)
    return None


def evaluate_checkpoint(
    checkpoint: str,
    eval_script: str,
    n_tta: int = 1,
    aggregation: str = "mean",
) -> Dict[str, float]:
    """Evaluate a checkpoint and return metrics."""
    import tempfile

    output_json = tempfile.mktemp(suffix=".json")
    cmd = [
        sys.executable, eval_script,
        "--checkpoint", checkpoint,
        "--tta", str(n_tta),
        "--aggregation", aggregation,
        "--output_json", output_json,
    ]

    try:
        subprocess.run(cmd, check=True)
        with open(output_json) as f:
            return json.load(f)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        logger.error("Evaluation failed for %s: %s", checkpoint, e)
        return {}
    finally:
        if os.path.isfile(output_json):
            os.remove(output_json)


def evaluate_precomputed(
    vjepa_ckpt: str, alt_ckpt: str,
) -> List[Dict[str, Any]]:
    """Evaluate two pre-trained checkpoints (skip training)."""
    eval_script = str(SCRIPT_DIR / "eval_e2e.py")
    results = []

    for label, ckpt in [("V-JEPA 2 (champion)", vjepa_ckpt), ("Alt backbone", alt_ckpt)]:
        logger.info("Evaluating: %s -> %s", label, ckpt)
        metrics = evaluate_checkpoint(ckpt, eval_script)
        results.append({
            "backbone": label,
            "checkpoint": ckpt,
            **metrics,
        })

    return results


def print_table(results: List[Dict[str, Any]]) -> None:
    """Print results in Table 5 format."""
    print()
    print("=" * 75)
    print("  Table 5: Cross-Backbone Transfer with Champion Hyperparameters")
    print("=" * 75)
    print(f"  {'Backbone':<40} {'mAP_ALL':>10} {'Delta':>10}")
    print("-" * 75)

    # Find champion mAP for delta computation
    champion_map = None
    for r in results:
        if "vjepa2-vitl" in r.get("backbone", "").lower() or "champion" in r.get("backbone", "").lower():
            champion_map = r.get("mAP_ALL", 0.0)
            break
    if champion_map is None and results:
        champion_map = results[0].get("mAP_ALL", 0.0)

    for r in results:
        backbone = r.get("backbone", "unknown")
        map_all = r.get("mAP_ALL", 0.0)
        delta = map_all - (champion_map or 0.0)
        delta_str = f"{delta:+.4f}" if champion_map else "—"
        print(f"  {backbone:<40} {map_all:>10.4f} {delta_str:>10}")

    print("=" * 75)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Cross-backbone transfer experiment (Table 5)",
    )

    # Mode 1: Train from config
    parser.add_argument(
        "--config", type=str, default=None,
        help="Champion config YAML (trains each backbone from scratch)",
    )
    parser.add_argument(
        "--backbones", type=str, nargs="+", default=None,
        help="List of backbone model names to compare",
    )
    parser.add_argument(
        "--work_dir", type=str, default="./xfer_runs",
        help="Working directory for training outputs",
    )

    # Mode 2: Evaluate pre-trained checkpoints
    parser.add_argument(
        "--vjepa", type=str, default=None,
        help="Pre-trained V-JEPA 2 champion checkpoint",
    )
    parser.add_argument(
        "--alt", type=str, default=None,
        help="Pre-trained alternative backbone checkpoint",
    )

    parser.add_argument(
        "--output", type=str, default="table5_results.json",
        help="Output JSON path for results",
    )
    args = parser.parse_args()

    results: List[Dict[str, Any]] = []

    if args.vjepa and args.alt:
        # Mode 2: Pre-trained checkpoints
        results = evaluate_precomputed(args.vjepa, args.alt)

    elif args.config:
        # Mode 1: Train from config with each backbone
        config = load_config(args.config)
        backbones = args.backbones or DEFAULT_BACKBONES
        train_script = str(SCRIPT_DIR / "train_vjepa2.py")
        eval_script = str(SCRIPT_DIR / "eval_e2e.py")

        for backbone in backbones:
            safe_name = backbone.replace("/", "_").replace("-", "_")
            run_dir = os.path.join(args.work_dir, safe_name)

            ckpt = train_with_backbone(config, backbone, run_dir, train_script)
            if ckpt is None:
                results.append({
                    "backbone": backbone,
                    "mAP_ALL": None,
                    "error": "training failed",
                })
                continue

            metrics = evaluate_checkpoint(ckpt, eval_script)
            results.append({
                "backbone": backbone,
                "checkpoint": ckpt,
                **metrics,
            })

    else:
        parser.error(
            "Provide either --config (train from scratch) or "
            "--vjepa + --alt (evaluate pre-trained checkpoints)"
        )

    # Print and save results
    print_table(results)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
