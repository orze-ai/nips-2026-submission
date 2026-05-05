#!/usr/bin/env python3
"""eval_e2e.py — End-to-end evaluation wrapper for Nexar collision prediction.

Runs TTA inference with the specified configuration and computes mAP metrics
(public, private, and combined mAP_ALL).

Usage:
    # Champion result (12-TTA + cv_mix, expected 0.910 mAP_ALL):
    python scripts/nexar/eval_e2e.py \
        --checkpoint /path/to/best_model.pt \
        --tta 12 --aggregation cv_mix

    # 4-TTA baseline (mean aggregation, expected 0.906 mAP_ALL):
    python scripts/nexar/eval_e2e.py \
        --checkpoint /path/to/best_model.pt \
        --tta 4 --aggregation mean
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent


def compute_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute average precision (AP) for binary classification."""
    sorted_indices = np.argsort(-y_score)
    y_true_sorted = y_true[sorted_indices]

    tp_cumsum = np.cumsum(y_true_sorted)
    n_positives = y_true_sorted.sum()
    if n_positives == 0:
        return 0.0

    precision = tp_cumsum / np.arange(1, len(y_true_sorted) + 1)
    recall_change = y_true_sorted / n_positives

    return float(np.sum(precision * recall_change))


def compute_map_splits(
    predictions: Dict[str, float],
    ground_truth: Dict[str, int],
    public_ids: Optional[set] = None,
    private_ids: Optional[set] = None,
) -> Dict[str, float]:
    """Compute mAP overall, and on public/private splits if provided."""
    common = sorted(set(predictions.keys()) & set(ground_truth.keys()))
    if not common:
        logger.warning("No overlap between predictions and ground truth")
        return {"mAP_ALL": 0.0}

    y_true = np.array([ground_truth[vid] for vid in common])
    y_score = np.array([predictions[vid] for vid in common])

    results = {"mAP_ALL": compute_average_precision(y_true, y_score)}

    if public_ids:
        pub_mask = np.array([vid in public_ids for vid in common])
        if pub_mask.any():
            results["mAP_public"] = compute_average_precision(
                y_true[pub_mask], y_score[pub_mask],
            )

    if private_ids:
        priv_mask = np.array([vid in private_ids for vid in common])
        if priv_mask.any():
            results["mAP_private"] = compute_average_precision(
                y_true[priv_mask], y_score[priv_mask],
            )

    return results


def load_ground_truth(
    csv_path: str,
) -> tuple[Dict[str, int], set, set]:
    """Load ground truth labels and public/private split IDs from CSV.

    Expected columns: video_id, label, [split].
    If 'split' column exists, returns public/private ID sets.
    """
    import csv

    gt: Dict[str, int] = {}
    public_ids: set = set()
    private_ids: set = set()

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("video_id", row.get("id", ""))
            label = int(row.get("label", row.get("target", 0)))
            gt[vid] = label

            split = row.get("split", "").lower()
            if split == "public":
                public_ids.add(vid)
            elif split == "private":
                private_ids.add(vid)

    return gt, public_ids, private_ids


def run_tta_inference(
    checkpoint: str,
    n_tta: int,
    aggregation: str,
    cv_mix_alpha: float,
    output_path: str,
    video_dir: str,
    test_csv: str,
    mean_pool: bool = False,
) -> str:
    """Call extract_tta_dense_end.py as a subprocess and return output .npz path."""
    script = str(SCRIPT_DIR / "extract_tta_dense_end.py")
    cmd = [
        sys.executable, script,
        "--ckpt", checkpoint,
        "--n_tta", str(n_tta),
        "--aggregation", aggregation,
        "--cv_mix_alpha", str(cv_mix_alpha),
        "--mean_pool", str(mean_pool),
        "--output", output_path,
        "--video_dir", video_dir,
        "--test_csv", test_csv,
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=True, capture_output=False)
    return output_path


def print_summary(metrics: Dict[str, float], tta: int, aggregation: str) -> None:
    """Print a formatted summary table."""
    print()
    print("=" * 60)
    print(f"  Nexar Collision Prediction — Evaluation Summary")
    print(f"  TTA views: {tta}  |  Aggregation: {aggregation}")
    print("=" * 60)
    print(f"  {'Metric':<20} {'Value':>10}")
    print("-" * 60)
    for key in ["mAP_ALL", "mAP_public", "mAP_private"]:
        if key in metrics:
            print(f"  {key:<20} {metrics[key]:>10.4f}")
    print("=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end evaluation for Nexar collision prediction",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained checkpoint (.pt)",
    )
    parser.add_argument(
        "--tta", type=int, default=12, choices=[1, 4, 12],
        help="Number of TTA views (1, 4, or 12)",
    )
    parser.add_argument(
        "--aggregation", type=str, default="mean",
        choices=["mean", "cv_mix"],
        help="Aggregation strategy: 'mean' (max over clips) or "
             "'cv_mix' (alpha*last + (1-alpha)*top6_mean)",
    )
    parser.add_argument(
        "--cv_mix_alpha", type=float, default=0.95,
        help="Alpha for cv_mix aggregation (default 0.95)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output .npz path (default: auto-generated temp file)",
    )
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="Save metrics as JSON to this path",
    )
    parser.add_argument(
        "--video_dir", type=str,
        default=os.environ.get(
            "TEST_VIDEO_DIR",
            "/workspace/datasets/nexar_collision/test",
        ),
        help="Directory containing test .mp4 files",
    )
    parser.add_argument(
        "--test_csv", type=str,
        default="/workspace/datasets/nexar_collision/test.csv",
        help="Test CSV with video_id, label, [split] columns",
    )
    parser.add_argument(
        "--mean_pool", action="store_true",
        help="Use mean pooling instead of attentive probe",
    )
    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        tmpdir = tempfile.mkdtemp(prefix="eval_e2e_")
        output_npz = os.path.join(tmpdir, "predictions.npz")
    else:
        output_npz = args.output

    # Step 1: Run TTA inference
    logger.info(
        "Starting evaluation: tta=%d, aggregation=%s, alpha=%.3f",
        args.tta, args.aggregation, args.cv_mix_alpha,
    )
    run_tta_inference(
        checkpoint=args.checkpoint,
        n_tta=args.tta,
        aggregation=args.aggregation,
        cv_mix_alpha=args.cv_mix_alpha,
        output_path=output_npz,
        video_dir=args.video_dir,
        test_csv=args.test_csv,
        mean_pool=args.mean_pool,
    )

    # Step 2: Load predictions
    data = np.load(output_npz)
    predictions = dict(zip(data["video_ids"], data["probabilities"]))
    logger.info("Loaded %d predictions from %s", len(predictions), output_npz)

    # Step 3: Load ground truth and compute mAP
    if os.path.isfile(args.test_csv):
        gt, public_ids, private_ids = load_ground_truth(args.test_csv)
        metrics = compute_map_splits(
            predictions, gt,
            public_ids=public_ids or None,
            private_ids=private_ids or None,
        )
    else:
        logger.warning(
            "Ground truth CSV not found at %s — skipping mAP computation. "
            "Predictions saved to %s",
            args.test_csv, output_npz,
        )
        metrics = {}

    # Step 4: Print results
    if metrics:
        print_summary(metrics, args.tta, args.aggregation)
    else:
        logger.info("Predictions saved to %s (no ground truth for mAP)", output_npz)

    # Step 5: Optionally save metrics
    if args.output_json and metrics:
        with open(args.output_json, "w") as f:
            json.dump(
                {
                    "tta": args.tta,
                    "aggregation": args.aggregation,
                    "cv_mix_alpha": args.cv_mix_alpha,
                    **metrics,
                },
                f, indent=2,
            )
        logger.info("Metrics saved to %s", args.output_json)


if __name__ == "__main__":
    main()
