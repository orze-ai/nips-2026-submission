#!/usr/bin/env python3
"""Reconstructed train_vjepa2.py — exports needed by chase925_tta_all12.py and score_checkpoint.py.

Original file was untracked in git and lost; this reconstruction captures the interface
as inferred from all surviving dependent scripts.
"""
from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
CROP_SIZE = 256
TEST_VIDEO_DIR = os.environ.get("TEST_VIDEO_DIR", "/workspace/datasets/nexar_collision/test")
TEST_CSV = "/workspace/datasets/nexar_collision/test.csv"

# Standard ImageNet normalisation (matches AutoVideoProcessor for vjepa2-vitl-fpc*-256)
VJEPA2_MEAN = [0.485, 0.456, 0.406]
VJEPA2_STD  = [0.229, 0.224, 0.225]

HF_MODEL_CACHE = "/workspace/cache/huggingface"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def decode_video_frames(video_path: str, frame_indices: List[int]) -> Dict[int, np.ndarray]:
    """Decode specific frames from an MP4 using OpenCV sequential read.

    Sequential read is far faster than random seeking (avoids GOP key-frame
    decoding overhead for each seek).  We scan forward from frame 0 to the
    last needed frame, saving only the requested indices.

    Returns a dict mapping frame index → RGB uint8 ndarray (H, W, 3).
    Missing frames are silently omitted.
    """
    if not frame_indices:
        return {}
    needed = set(int(i) for i in frame_indices)
    max_idx = max(needed)

    cap = cv2.VideoCapture(str(video_path))
    result: Dict[int, np.ndarray] = {}
    current = 0
    while current <= max_idx:
        ret, frame = cap.read()
        if not ret:
            break
        if current in needed:
            result[current] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if len(result) == len(needed):
                break  # got everything we need
        current += 1
    cap.release()
    return result


def get_video_info(video_path: str) -> Tuple[int, float]:
    """Return (n_frames, fps) for a video file."""
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return n_frames, fps


def load_test_annotations(csv_path: str = TEST_CSV) -> List[dict]:
    """Read test CSV and return list of row dicts."""
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(
    model_name: str = "facebook/vjepa2-vitl-fpc64-256",
    num_frames: int = 16,
    num_probe_queries: int = 1,
    use_mlp_head: bool = False,
    mean_pool: bool = False,
    drop_path_rate: float = 0.0,
) -> nn.Module:
    """Build V-JEPA 2 ViT-L with a binary classification head.

    Architecture matches champion training:
      - VJEPA2ForVideoClassification (attentive pooler + linear head)
      - num_labels=1 (binary: collision vs non-collision)
      - frames_per_clip=num_frames (for positional embedding size)

    Args:
        model_name:        HuggingFace model id (backbone config source).
        num_frames:        Number of input frames per clip (16 for champion).
        num_probe_queries: Attentive pooler queries (1 for champion).
        use_mlp_head:      If True, use 2-layer MLP classifier instead of linear.
        mean_pool:         If True, replace attentive pooler with mean pooling.
        drop_path_rate:    Stochastic depth rate (0 for inference).

    Returns:
        nn.Module with forward(pixel_values_videos) → ImageClassifierOutput(.logits)
    """
    if os.path.isdir(HF_MODEL_CACHE):
        os.environ.setdefault("HF_HOME", HF_MODEL_CACHE)

    from transformers import VJEPA2Config
    from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2ForVideoClassification

    config = VJEPA2Config.from_pretrained(model_name)
    config.num_labels = 1
    config.frames_per_clip = num_frames      # positional embeddings sized for num_frames
    config.drop_path_rate = drop_path_rate

    model = VJEPA2ForVideoClassification(config)

    if mean_pool:
        # Replace attentive pooler with simple mean over sequence.
        # keepdim=True so classifier still sees (B, 1, D) → logits (B, 1, 1).
        model.pooler.forward = lambda h: h.mean(dim=1, keepdim=True)

    if use_mlp_head:
        D = config.hidden_size
        model.classifier = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, config.num_labels),
        )

    return model


# ---------------------------------------------------------------------------
# Dataset (used by score_checkpoint.py)
# ---------------------------------------------------------------------------

class VideoEvalDataset(torch.utils.data.Dataset):
    """Clip-level dataset for offline evaluation.

    Slides a window of ``clip_frames`` over every video at ``sample_stride``
    step.  Each item is (clip_tensor, video_id) where clip_tensor has shape
    (T, 3, H, W) and is normalised with VJEPA2_MEAN/STD.
    """

    def __init__(
        self,
        video_ids: List[str],
        video_dir: str,
        sample_stride: int = 30,
        clip_frames: int = 16,
        clip_stride: int = 4,
        crop_size: int = CROP_SIZE,
    ) -> None:
        self.clip_frames = clip_frames
        self.clip_stride = clip_stride
        self.crop_size = crop_size
        self.clips: List[Tuple[str, str, int, int]] = []

        for vid in video_ids:
            vpath = os.path.join(video_dir, f"{vid}.mp4")
            n_frames, _ = get_video_info(vpath)
            if n_frames == 0:
                continue
            for center in range(0, n_frames, sample_stride):
                self.clips.append((vid, vpath, int(center), int(n_frames)))

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int):
        vid, vpath, center, n_total = self.clips[idx]

        frame_indices = []
        for i in range(self.clip_frames):
            fi = center - (self.clip_frames - 1 - i) * self.clip_stride
            fi = max(0, min(int(fi), n_total - 1))
            frame_indices.append(fi)

        decoded = decode_video_frames(vpath, sorted(set(frame_indices)))

        blank = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
        frames = np.stack(
            [_center_crop(decoded.get(fi, blank), self.crop_size) for fi in frame_indices],
            axis=0,
        )  # (T, H, W, 3)

        clip = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        mean_t = torch.tensor(VJEPA2_MEAN).view(1, 3, 1, 1)
        std_t  = torch.tensor(VJEPA2_STD).view(1, 3, 1, 1)
        clip = (clip - mean_t) / std_t

        return clip, vid


def _center_crop(frame: np.ndarray, size: int) -> np.ndarray:
    """Resize shortest side to `size`, then center-crop to size×size."""
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    if h < w:
        new_h, new_w = size, int(w * size / h)
    else:
        new_h, new_w = int(h * size / w), size
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top  = (new_h - size) // 2
    left = (new_w - size) // 2
    return resized[top:top + size, left:left + size]
