#!/usr/bin/env python3
"""extract_tta_dense_end.py — Test-Time Augmentation inference for Nexar collision.

Reconstruction note
-------------------
This is a faithful reconstruction of the 12-TTA inference pipeline described
in the paper.  Given a trained checkpoint, it applies N augmentation views
(horizontal flip, frame stride variations, spatial crops) to each test video,
aggregates per-view sigmoid probabilities, and saves the result as an .npz.

Usage:
    python scripts/nexar/extract_tta_dense_end.py \
        --ckpt /path/to/best_model.pt \
        --n_tta 12 \
        --mean_pool False
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

# Allow importing from the same package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nexar.train_vjepa2 import (
    CROP_SIZE,
    VJEPA2_MEAN,
    VJEPA2_STD,
    VideoEvalDataset,
    _center_crop,
    _horizontal_flip,
    build_model,
    decode_video_frames,
    get_video_info,
    load_test_annotations,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TTA View Definitions
# ---------------------------------------------------------------------------

# Each view is a dict with:
#   hflip: bool        — apply horizontal flip
#   stride: int        — clip_stride override (default 4)
#   crop: str          — spatial crop variant (center/left/right/top/bottom)

TTA_VIEWS_12 = [
    {"hflip": False, "stride": 4, "crop": "center"},       # 0: identity
    {"hflip": True,  "stride": 4, "crop": "center"},       # 1: hflip
    {"hflip": False, "stride": 3, "crop": "center"},       # 2: fast stride
    {"hflip": False, "stride": 5, "crop": "center"},       # 3: slow stride
    {"hflip": False, "stride": 4, "crop": "left"},          # 4: left crop
    {"hflip": False, "stride": 4, "crop": "right"},         # 5: right crop
    {"hflip": False, "stride": 4, "crop": "top"},           # 6: top crop
    {"hflip": False, "stride": 4, "crop": "bottom"},        # 7: bottom crop
    {"hflip": True,  "stride": 3, "crop": "center"},       # 8: hflip + fast
    {"hflip": True,  "stride": 5, "crop": "center"},       # 9: hflip + slow
    {"hflip": True,  "stride": 4, "crop": "left"},          # 10: hflip + left
    {"hflip": True,  "stride": 4, "crop": "right"},         # 11: hflip + right
]


def _spatial_crop(frame: np.ndarray, crop: str, size: int) -> np.ndarray:
    """Resize shortest side to `size`, then crop according to `crop` mode."""
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    if h < w:
        new_h, new_w = size, int(w * size / h)
    else:
        new_h, new_w = int(h * size / w), size
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if crop == "center":
        top = (new_h - size) // 2
        left = (new_w - size) // 2
    elif crop == "left":
        top = (new_h - size) // 2
        left = 0
    elif crop == "right":
        top = (new_h - size) // 2
        left = max(0, new_w - size)
    elif crop == "top":
        top = 0
        left = (new_w - size) // 2
    elif crop == "bottom":
        top = max(0, new_h - size)
        left = (new_w - size) // 2
    else:
        top = (new_h - size) // 2
        left = (new_w - size) // 2

    return resized[top : top + size, left : left + size]


def prepare_clip(
    video_path: str,
    center_frame: int,
    n_total: int,
    clip_frames: int,
    clip_stride: int,
    crop_size: int,
    crop_mode: str,
    hflip: bool,
) -> torch.Tensor:
    """Decode and preprocess a single clip for one TTA view.

    Returns tensor of shape (T, 3, H, W), normalised.
    """
    frame_indices = []
    for i in range(clip_frames):
        fi = center_frame - (clip_frames - 1 - i) * clip_stride
        fi = max(0, min(int(fi), max(n_total - 1, 0)))
        frame_indices.append(fi)

    decoded = decode_video_frames(video_path, sorted(set(frame_indices)))
    blank = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)

    frames = np.stack(
        [_spatial_crop(decoded.get(fi, blank), crop_mode, crop_size)
         for fi in frame_indices],
        axis=0,
    )  # (T, H, W, 3)

    if hflip:
        frames = _horizontal_flip(frames)

    clip = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    mean_t = torch.tensor(VJEPA2_MEAN).view(1, 3, 1, 1)
    std_t = torch.tensor(VJEPA2_STD).view(1, 3, 1, 1)
    clip = (clip - mean_t) / std_t
    return clip


# ---------------------------------------------------------------------------
# Main Inference
# ---------------------------------------------------------------------------

def _aggregate_mean(clip_probs: List[float]) -> float:
    """Default aggregation: max over clip positions (dense-end)."""
    return float(max(clip_probs)) if clip_probs else 0.0


def _aggregate_cv_mix(
    clip_probs: List[float], alpha: float = 0.95,
) -> float:
    """CV-mix aggregation: alpha * last + (1-alpha) * top6_mean.

    - `last` = prediction from the last temporal clip (highest index).
    - `top6_mean` = mean of the top-6 predictions by confidence score.
    - Final = alpha * last + (1 - alpha) * top6_mean.

    This aggregation leverages the observation that the final clip captures
    the collision moment, while the top-6 mean provides a robust confidence
    baseline.  alpha=0.95 was tuned on the public validation split.
    """
    if not clip_probs:
        return 0.0
    last = clip_probs[-1]  # last temporal position
    sorted_desc = sorted(clip_probs, reverse=True)
    top6 = sorted_desc[:min(6, len(sorted_desc))]
    top6_mean = float(np.mean(top6))
    return float(alpha * last + (1.0 - alpha) * top6_mean)


@torch.no_grad()
def run_tta_inference(
    model: nn.Module,
    video_ids: List[str],
    video_dir: str,
    n_tta: int = 12,
    clip_frames: int = 16,
    crop_size: int = CROP_SIZE,
    sample_stride: int = 30,
    batch_size: int = 16,
    device: torch.device = torch.device("cpu"),
    aggregation: str = "mean",
    cv_mix_alpha: float = 0.95,
) -> Dict[str, float]:
    """Run TTA inference over all test videos.

    For each video, slides a dense window and applies `n_tta` augmentation
    views. Per-clip sigmoid probabilities are averaged across views, then
    the per-video score is determined by the chosen aggregation strategy:

    - ``mean`` (default): max over clip positions (dense-end protocol).
    - ``cv_mix``: alpha * last_clip + (1 - alpha) * top6_mean, where
      alpha is tuned on the public split (default 0.95).

    Returns dict mapping video_id -> collision probability.
    """
    model.eval()
    views = TTA_VIEWS_12[:n_tta]
    results: Dict[str, List[float]] = {vid: [] for vid in video_ids}

    for vid_idx, vid in enumerate(video_ids):
        vpath = os.path.join(video_dir, f"{vid}.mp4")
        if not os.path.isfile(vpath):
            logger.warning("Video not found: %s", vpath)
            continue

        n_frames, fps = get_video_info(vpath)
        if n_frames == 0:
            continue

        # Dense sampling: slide over video at sample_stride
        centers = list(range(0, n_frames, sample_stride))

        for center in centers:
            view_probs = []
            for view in views:
                clip = prepare_clip(
                    vpath, center, n_frames,
                    clip_frames=clip_frames,
                    clip_stride=view["stride"],
                    crop_size=crop_size,
                    crop_mode=view["crop"],
                    hflip=view["hflip"],
                )
                clip = clip.unsqueeze(0).to(device)  # (1, T, 3, H, W)
                out = model(pixel_values_videos=clip)
                logit = out.logits.squeeze().cpu().item()
                prob = 1.0 / (1.0 + np.exp(-logit))  # sigmoid
                view_probs.append(prob)

            # Aggregate across views: arithmetic mean
            avg_prob = float(np.mean(view_probs))
            results[vid].append(avg_prob)

        if (vid_idx + 1) % 50 == 0:
            logger.info("Processed %d/%d videos", vid_idx + 1, len(video_ids))

    # Per-video score: aggregate clip-level predictions
    final: Dict[str, float] = {}
    for vid, probs in results.items():
        if aggregation == "cv_mix":
            final[vid] = _aggregate_cv_mix(probs, alpha=cv_mix_alpha)
        else:
            final[vid] = _aggregate_mean(probs)

    return final


def main():
    parser = argparse.ArgumentParser(
        description="TTA inference for Nexar collision prediction",
    )
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Path to trained checkpoint (.pt)",
    )
    parser.add_argument(
        "--n_tta", type=int, default=12,
        help="Number of TTA views (max 12)",
    )
    parser.add_argument(
        "--mean_pool", type=str, default="False",
        help="Use mean pooling instead of attentive probe (True/False)",
    )
    parser.add_argument(
        "--output", type=str, default="tta_predictions.npz",
        help="Output .npz file path",
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
        help="Test CSV with video_id column",
    )
    parser.add_argument(
        "--aggregation", type=str, default="mean",
        choices=["mean", "cv_mix"],
        help="Clip-level aggregation: 'mean' = max over clips (default), "
             "'cv_mix' = alpha*last + (1-alpha)*top6_mean",
    )
    parser.add_argument(
        "--cv_mix_alpha", type=float, default=0.95,
        help="Alpha for cv_mix aggregation (default 0.95, tuned on public split)",
    )
    parser.add_argument("--clip_frames", type=int, default=16)
    parser.add_argument("--crop_size", type=int, default=CROP_SIZE)
    parser.add_argument("--sample_stride", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean_pool = args.mean_pool.lower() in ("true", "1", "yes")

    # Load checkpoint
    logger.info("Loading checkpoint: %s", args.ckpt)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # Recover config from checkpoint if available
    ckpt_cfg = ckpt.get("config", {})
    model_name = ckpt_cfg.get("model_name", "facebook/vjepa2-vitl-fpc64-256")

    model = build_model(
        model_name=model_name,
        num_frames=args.clip_frames,
        mean_pool=mean_pool,
        drop_path_rate=0.0,  # no stochastic depth at inference
    )

    # Load state dict (handles both raw state_dict and wrapped checkpoint)
    state_dict = ckpt.get("model_state_dict", ckpt)
    if isinstance(state_dict, dict) and not any(
        k.startswith("model.") or k.startswith("classifier.")
        for k in state_dict
    ):
        # Might be the full checkpoint object itself
        pass
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    logger.info(
        "Model loaded (mean_pool=%s, n_tta=%d, aggregation=%s)",
        mean_pool, args.n_tta, args.aggregation,
    )

    # Get test video IDs
    annotations = load_test_annotations(args.test_csv)
    video_ids = sorted(set(
        r.get("video_id", r.get("id", "")) for r in annotations
    ))
    logger.info("Test videos: %d", len(video_ids))

    # Run TTA inference
    predictions = run_tta_inference(
        model=model,
        video_ids=video_ids,
        video_dir=args.video_dir,
        n_tta=args.n_tta,
        clip_frames=args.clip_frames,
        crop_size=args.crop_size,
        sample_stride=args.sample_stride,
        batch_size=args.batch_size,
        device=device,
        aggregation=args.aggregation,
        cv_mix_alpha=args.cv_mix_alpha,
    )

    # Save as .npz
    np.savez(
        args.output,
        video_ids=np.array(list(predictions.keys())),
        probabilities=np.array(list(predictions.values())),
    )
    logger.info(
        "Saved %d predictions to %s", len(predictions), args.output,
    )


if __name__ == "__main__":
    main()
