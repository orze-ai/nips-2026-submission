#!/usr/bin/env python3
"""Standalone SMAC3 search on the core Nexar collision-prediction schema.

Reproduces the "SMAC on core schema" baseline from Table 6 / Table 7.
Uses SMAC3's multi-fidelity Hyperband facade (SMAC-HB) with the same
configuration space as Definition 1 (core subspace):

  C^core = {5 backbones} x {4 encoders} x {2 poolings}
         x continuous HPs (lr, weight_decay, focal_gamma, focal_alpha, ...)

Each trial trains a frozen-feature model via train_vjepa2.py (or the
generic training entrypoint) and returns validation mAP.

Requirements:
    pip install smac>=2.0 ConfigSpace

Usage:
    python scripts/dashcam_risk/run_smac_search.py \
        --n-trials 569 --seed 0 --output-dir results/smac_core

    # Multiple seeds (paper uses 8):
    for s in 0 1 2 3 4 5 6 7; do
        python scripts/dashcam_risk/run_smac_search.py \
            --n-trials 72 --seed $s --output-dir results/smac_core_seed$s
    done
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

try:
    from ConfigSpace import (
        Categorical,
        ConfigurationSpace,
        Float,
        Integer,
    )
except ImportError:
    sys.exit("ConfigSpace not installed. Run: pip install ConfigSpace>=0.7")

try:
    from smac import HyperbandFacade, Scenario
except ImportError:
    sys.exit("SMAC3 not installed. Run: pip install smac>=2.0")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "nexar" / "train_vjepa2.py"

# ---------------------------------------------------------------------------
# Core schema (Definition 1, C^core): 5 backbones x 4 encoders x 2 poolings
# ---------------------------------------------------------------------------
CORE_BACKBONES = [
    "facebook/dinov2-base",           # DINOv2-B
    "timm/vit_base_patch16_dinov3.lvd1689m",  # DINOv2-B-reg
    "timm/vit_large_patch14_dinov2.lvd142m",  # DINOv2-L-reg
    "google/siglip2-base-patch16-256",         # SigLIP2
    "OpenGVLab/InternViT-300M-448px",          # InternViT
]

CORE_ENCODERS = [
    "zipformer",
    "retnet",
    "bimamba",
    "transformer",
]

CORE_POOLINGS = [
    "attention",
    "mean",
]

LOSS_TYPES = ["focal", "bce", "label_smoothing"]


def build_configspace(seed: int = 0) -> ConfigurationSpace:
    """Build the core configuration space matching Definition 1."""
    cs = ConfigurationSpace(seed=seed)

    cs.add(Categorical("backbone", CORE_BACKBONES))
    cs.add(Categorical("encoder", CORE_ENCODERS))
    cs.add(Categorical("pooling", CORE_POOLINGS))

    cs.add(Float("lr", (1e-5, 1e-2), log=True))
    cs.add(Float("weight_decay", (1e-6, 0.1), log=True))
    cs.add(Integer("batch_size", (8, 64)))
    cs.add(Integer("clip_frames", (8, 32)))
    cs.add(Integer("epochs", (10, 50)))

    cs.add(Categorical("loss_type", LOSS_TYPES))
    cs.add(Float("focal_gamma", (0.5, 5.0)))
    cs.add(Float("focal_alpha", (0.1, 0.9)))

    cs.add(Float("mixup_alpha", (0.0, 1.0)))
    cs.add(Float("drop_path_rate", (0.0, 0.3)))
    cs.add(Float("label_smoothing", (0.0, 0.2)))

    return cs


def train_and_evaluate(config, seed: int = 0, budget: float = 1.0) -> float:
    """Train a model with the given config and return 1 - val_mAP (SMAC minimizes).

    Parameters
    ----------
    config : Configuration
        SMAC configuration with HP values.
    seed : int
        Random seed for reproducibility.
    budget : float
        Fraction of max epochs to train (for multi-fidelity).

    Returns
    -------
    float
        1 - val_mAP (lower is better for SMAC).
    """
    epochs = max(1, int(config["epochs"] * budget))

    with tempfile.TemporaryDirectory(prefix="smac_trial_") as tmpdir:
        config_dict = {
            "model_name": config["backbone"],
            "freeze_backbone": True,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_targets": ["q_proj", "v_proj"],
            "num_probe_queries": 1,
            "mean_pool": config["pooling"] == "mean",
            "temporal_encoder": config["encoder"],
            "crop_size": 256,
            "clip_frames": config["clip_frames"],
            "clip_stride": 4,
            "epochs": epochs,
            "batch_size": config["batch_size"],
            "lr": float(config["lr"]),
            "weight_decay": float(config["weight_decay"]),
            "mixup_alpha": float(config["mixup_alpha"]),
            "drop_path_rate": float(config["drop_path_rate"]),
            "label_smoothing": float(config["label_smoothing"]),
            "loss_type": config["loss_type"],
            "focal_gamma": float(config["focal_gamma"]),
            "focal_alpha": float(config["focal_alpha"]),
            "sam_enabled": False,
            "save_dir": tmpdir,
            "eval_metric": "pgmAP_ALL",
            "seed": seed,
        }

        config_path = Path(tmpdir) / "config.yaml"
        try:
            import yaml
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f)
        except ImportError:
            with open(config_path, "w") as f:
                json.dump(config_dict, f)

        cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "--config", str(config_path),
        ]

        logger.info(
            "Trial: backbone=%s encoder=%s pooling=%s lr=%.2e epochs=%d",
            config["backbone"], config["encoder"], config["pooling"],
            config["lr"], epochs,
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Trial timed out after 7200s")
            return 1.0

        metrics_path = Path(tmpdir) / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            val_map = metrics.get("best_val_metric", 0.0)
            logger.info("  -> val_mAP = %.4f", val_map)
            return 1.0 - val_map

        for line in reversed(result.stdout.strip().split("\n")[-20:]):
            if "best_val_metric" in line or "pgmAP" in line:
                try:
                    val_map = float(line.split("=")[-1].strip().rstrip(","))
                    logger.info("  -> val_mAP = %.4f (from stdout)", val_map)
                    return 1.0 - val_map
                except ValueError:
                    pass

        logger.warning("  -> Trial failed, returning 1.0")
        if result.stderr:
            logger.warning("  stderr: %s", result.stderr[-500:])
        return 1.0


def main():
    parser = argparse.ArgumentParser(
        description="SMAC3 search on core Nexar schema (Table 6/7 reproduction)"
    )
    parser.add_argument("--n-trials", type=int, default=569,
                        help="Number of trials (paper: 569)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results/smac_search")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel workers (requires Dask)")
    parser.add_argument("--walltime-limit", type=int, default=None,
                        help="Walltime limit in seconds (default: no limit)")
    parser.add_argument("--min-budget", type=float, default=0.1,
                        help="Minimum budget fraction for Hyperband")
    parser.add_argument("--max-budget", type=float, default=1.0,
                        help="Maximum budget fraction for Hyperband")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cs = build_configspace(seed=args.seed)

    scenario = Scenario(
        configspace=cs,
        deterministic=False,
        n_trials=args.n_trials,
        seed=args.seed,
        output_directory=output_dir / "smac_output",
        walltime_limit=args.walltime_limit or np.inf,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
        n_workers=args.n_workers,
    )

    smac = HyperbandFacade(
        scenario=scenario,
        target_function=train_and_evaluate,
        overwrite=True,
    )

    logger.info("Starting SMAC search: %d trials, seed=%d", args.n_trials, args.seed)
    logger.info("Core schema: %d backbones x %d encoders x %d poolings",
                len(CORE_BACKBONES), len(CORE_ENCODERS), len(CORE_POOLINGS))
    logger.info("Output: %s", output_dir)

    incumbent = smac.optimize()

    best_cost = smac.validate(incumbent)
    best_map = 1.0 - best_cost

    logger.info("=" * 60)
    logger.info("SMAC search complete")
    logger.info("Best config: %s", dict(incumbent))
    logger.info("Best val mAP: %.4f", best_map)
    logger.info("=" * 60)

    summary = {
        "n_trials": args.n_trials,
        "seed": args.seed,
        "best_val_mAP": float(best_map),
        "best_config": dict(incumbent),
        "schema": "core",
        "backbones": CORE_BACKBONES,
        "encoders": CORE_ENCODERS,
        "poolings": CORE_POOLINGS,
    }

    summary_path = output_dir / "smac_search_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
