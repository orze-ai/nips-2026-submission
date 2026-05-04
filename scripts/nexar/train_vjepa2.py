#!/usr/bin/env python3
"""train_vjepa2.py — V-JEPA 2 collision-prediction training script.

Reconstruction note
-------------------
The original training script was untracked and lost. This is a faithful
reconstruction based on the paper's described optimization trajectory
(V-JEPA 2 ViT-L, attentive probe, LoRA r=16, alert-only filtering,
group-aligned BCE, Mixup α=0.2, SAM optimizer, DropPath). The public
API surface (build_model, VideoEvalDataset, decode_video_frames, set_seed)
is preserved for compatibility with downstream inference scripts.

Champion config: configs/nexar/champion/vjepa2_alertonly_v4.yaml
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import yaml
except ImportError:
    yaml = None  # graceful fallback — config can be passed via CLI flags

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
CROP_SIZE = 256
TEST_VIDEO_DIR = os.environ.get("TEST_VIDEO_DIR", "/workspace/datasets/nexar_collision/test")
TEST_CSV = "/workspace/datasets/nexar_collision/test.csv"

# Standard ImageNet normalisation (matches AutoVideoProcessor for vjepa2-vitl-fpc*-256)
VJEPA2_MEAN = [0.485, 0.456, 0.406]
VJEPA2_STD = [0.229, 0.224, 0.225]

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
    top = (new_h - size) // 2
    left = (new_w - size) // 2
    return resized[top : top + size, left : left + size]


def _random_crop(frame: np.ndarray, size: int) -> np.ndarray:
    """Resize shortest side to `size`, then random-crop to size×size."""
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    if h < w:
        new_h, new_w = size, int(w * size / h)
    else:
        new_h, new_w = int(h * size / w), size
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top = random.randint(0, max(0, new_h - size))
    left = random.randint(0, max(0, new_w - size))
    return resized[top : top + size, left : left + size]


def _horizontal_flip(frames: np.ndarray) -> np.ndarray:
    """Flip (T, H, W, 3) horizontally."""
    return frames[:, :, ::-1, :].copy()


# ---------------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------------

class Mixup:
    """Mixup augmentation for video classification (operates on batched tensors)."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.alpha <= 0:
            return x, y
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1 - lam)  # ensure lam >= 0.5 for stability
        idx = torch.randperm(x.size(0), device=x.device)
        x_mixed = lam * x + (1 - lam) * x[idx]
        y_mixed = lam * y + (1 - lam) * y[idx]
        return x_mixed, y_mixed


# ---------------------------------------------------------------------------
# SAM Optimizer
# ---------------------------------------------------------------------------

class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization wrapper.

    Wraps any base optimizer.  Each step performs two forward-backward passes:
    1. Perturb weights by epsilon in the gradient direction (ascent step)
    2. Compute gradient at perturbed point and perform the actual update (descent)

    Reference: Foret et al., "Sharpness-Aware Minimization for Efficiently
    Improving Generalization", ICLR 2021.
    """

    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **kwargs):
        defaults = dict(rho=rho)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)

    @torch.no_grad()
    def first_step(self):
        """Ascent step: perturb parameters toward steepest gradient direction."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self):
        """Descent step: restore parameters and apply base optimizer update."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # undo perturbation
        self.base_optimizer.step()

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack(
                [
                    p.grad.norm(p=2).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.base_optimizer.param_groups = value


# ---------------------------------------------------------------------------
# Group-Aligned BCE Loss
# ---------------------------------------------------------------------------

class GroupAlignedBCELoss(nn.Module):
    """Binary cross-entropy with per-group (video) loss balancing.

    Standard BCE on imbalanced clip-level labels biases toward the majority
    class within each video. Group-aligned CE reweights so that every video
    contributes equally regardless of how many positive/negative clips it has.
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        group_ids: Optional[List[str]] = None,
    ) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        per_sample_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        if group_ids is None:
            return per_sample_loss.mean()

        # Compute per-group mean, then average across groups
        group_to_indices = defaultdict(list)
        for i, gid in enumerate(group_ids):
            group_to_indices[gid].append(i)

        group_losses = []
        for indices in group_to_indices.values():
            group_losses.append(per_sample_loss[indices].mean())

        return torch.stack(group_losses).mean()


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
    config.frames_per_clip = num_frames  # positional embeddings sized for num_frames
    config.drop_path_rate = drop_path_rate

    model = VJEPA2ForVideoClassification(config)

    if mean_pool:
        # Replace attentive pooler with simple mean over sequence.
        model.pooler.forward = lambda h: h.mean(dim=1, keepdim=True)

    if use_mlp_head:
        D = config.hidden_size
        model.classifier = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, config.num_labels),
        )

    return model


def apply_lora(model: nn.Module, r: int = 16, alpha: int = 32,
               target_modules: Optional[List[str]] = None) -> nn.Module:
    """Freeze backbone and attach LoRA adapters to attention projections."""
    if LoraConfig is None or get_peft_model is None:
        raise ImportError("peft is required for LoRA. Install with: pip install peft")
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    # Freeze all backbone parameters
    for name, param in model.named_parameters():
        if "classifier" not in name and "pooler" not in name:
            param.requires_grad = False
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    logger.info(
        "LoRA applied (r=%d, α=%d). Trainable params: %s",
        r, alpha,
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
    )
    return model


# ---------------------------------------------------------------------------
# Training Dataset
# ---------------------------------------------------------------------------

class VideoTrainDataset(Dataset):
    """Clip-level dataset for training with alert-only filtering.

    Each sample is one clip of ``clip_frames`` frames centered on a labelled
    event timestamp. When ``alert_only=True``, only rows where the alert
    column indicates collision-relevant frames are kept.
    """

    def __init__(
        self,
        csv_path: str,
        video_dir: str,
        clip_frames: int = 16,
        clip_stride: int = 4,
        crop_size: int = CROP_SIZE,
        alert_only: bool = False,
        group_column: str = "video_id",
        augment: bool = True,
    ) -> None:
        self.clip_frames = clip_frames
        self.clip_stride = clip_stride
        self.crop_size = crop_size
        self.augment = augment
        self.group_column = group_column

        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

        if alert_only:
            # Keep only rows flagged as collision-relevant (alert / near-miss)
            before = len(rows)
            rows = [
                r for r in rows
                if r.get("alert", r.get("is_alert", "0")).strip() in ("1", "true", "True", "yes")
                or r.get("label", "0").strip() == "1"  # always keep positives
            ]
            logger.info("Alert-only filter: %d → %d clips", before, len(rows))

        self.samples: List[dict] = []
        for r in rows:
            vid = r.get("video_id", r.get("id", ""))
            vpath = os.path.join(video_dir, f"{vid}.mp4")
            if not os.path.isfile(vpath):
                continue
            self.samples.append({
                "video_id": vid,
                "video_path": vpath,
                "label": float(r.get("label", r.get("collision", 0))),
                "center_frame": int(r.get("center_frame", r.get("frame", 0))),
            })

        logger.info("VideoTrainDataset: %d samples from %s", len(self.samples), csv_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        n_frames, _ = get_video_info(s["video_path"])
        center = s["center_frame"]

        frame_indices = []
        for i in range(self.clip_frames):
            fi = center - (self.clip_frames - 1 - i) * self.clip_stride
            fi = max(0, min(int(fi), max(n_frames - 1, 0)))
            frame_indices.append(fi)

        decoded = decode_video_frames(s["video_path"], sorted(set(frame_indices)))
        blank = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)

        crop_fn = _random_crop if self.augment else _center_crop
        frames = np.stack(
            [crop_fn(decoded.get(fi, blank), self.crop_size) for fi in frame_indices],
            axis=0,
        )  # (T, H, W, 3)

        if self.augment and random.random() > 0.5:
            frames = _horizontal_flip(frames)

        clip = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        mean_t = torch.tensor(VJEPA2_MEAN).view(1, 3, 1, 1)
        std_t = torch.tensor(VJEPA2_STD).view(1, 3, 1, 1)
        clip = (clip - mean_t) / std_t

        label = torch.tensor(s["label"], dtype=torch.float32)
        return clip, label, s["video_id"]


# ---------------------------------------------------------------------------
# Eval Dataset (used by score_checkpoint.py)
# ---------------------------------------------------------------------------

class VideoEvalDataset(Dataset):
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
        std_t = torch.tensor(VJEPA2_STD).view(1, 3, 1, 1)
        clip = (clip - mean_t) / std_t

        return clip, vid


# ---------------------------------------------------------------------------
# Learning Rate Scheduler (cosine with warmup)
# ---------------------------------------------------------------------------

def cosine_lr_schedule(optimizer, epoch: int, max_epochs: int,
                       warmup_epochs: int, base_lr: float, min_lr: float = 1e-6):
    """Set learning rate with linear warmup + cosine decay."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> Tuple[float, float]:
    """Run validation and return (val_loss, val_auc).

    val_auc is clip-level AUC — a proxy for pgmAP_ALL.
    """
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    n_batches = 0

    for clips, labels, group_ids in val_loader:
        clips = clips.to(device)
        labels = labels.to(device)
        out = model(pixel_values_videos=clips)
        logits = out.logits.squeeze(-1).squeeze(-1)
        loss = criterion(logits, labels, list(group_ids))
        total_loss += loss.item()
        n_batches += 1
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    avg_loss = total_loss / max(n_batches, 1)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Clip-level AUC as selection proxy
    try:
        from sklearn.metrics import roc_auc_score
        probs = torch.sigmoid(all_logits).numpy()
        auc = roc_auc_score(all_labels.numpy(), probs)
    except Exception:
        auc = 0.0

    model.train()
    return avg_loss, auc


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.get("seed", SEED))

    # ── Model ──────────────────────────────────────────────────────────────
    logger.info("Building model: %s", cfg["model_name"])
    model = build_model(
        model_name=cfg["model_name"],
        num_frames=cfg.get("clip_frames", 16),
        num_probe_queries=cfg.get("num_probe_queries", 1),
        use_mlp_head=cfg.get("use_mlp_head", False),
        mean_pool=cfg.get("mean_pool", False),
        drop_path_rate=cfg.get("drop_path_rate", 0.1),
    )

    if cfg.get("freeze_backbone", False) and cfg.get("lora_r", 0) > 0:
        model = apply_lora(
            model,
            r=cfg["lora_r"],
            alpha=cfg.get("lora_alpha", 32),
            target_modules=cfg.get("lora_targets"),
        )
    elif cfg.get("freeze_backbone", False):
        for name, param in model.named_parameters():
            if "classifier" not in name and "pooler" not in name:
                param.requires_grad = False
        logger.info("Backbone frozen (no LoRA). Trainable: classifier + pooler only.")

    model = model.to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %s", f"{n_trainable:,}")

    # ── Data ───────────────────────────────────────────────────────────────
    train_ds = VideoTrainDataset(
        csv_path=cfg["train_csv"],
        video_dir=cfg["video_dir"],
        clip_frames=cfg.get("clip_frames", 16),
        clip_stride=cfg.get("clip_stride", 4),
        crop_size=cfg.get("crop_size", CROP_SIZE),
        alert_only=cfg.get("alert_only", False),
        group_column=cfg.get("group_column", "video_id"),
        augment=True,
    )
    val_ds = VideoTrainDataset(
        csv_path=cfg["val_csv"],
        video_dir=cfg["video_dir"],
        clip_frames=cfg.get("clip_frames", 16),
        clip_stride=cfg.get("clip_stride", 4),
        crop_size=cfg.get("crop_size", CROP_SIZE),
        alert_only=False,  # validate on all clips
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 16),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get("batch_size", 16),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )

    # ── Loss ───────────────────────────────────────────────────────────────
    criterion = GroupAlignedBCELoss(
        label_smoothing=cfg.get("label_smoothing", 0.0),
    ) if cfg.get("group_aligned_ce", True) else nn.BCEWithLogitsLoss()

    # ── Optimizer ──────────────────────────────────────────────────────────
    lr = cfg.get("lr", 3e-4)
    wd = cfg.get("weight_decay", 0.05)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if cfg.get("sam_enabled", False):
        optimizer = SAM(
            trainable_params,
            base_optimizer_cls=torch.optim.AdamW,
            rho=cfg.get("sam_rho", 0.05),
            lr=lr,
            weight_decay=wd,
        )
        logger.info("Using SAM optimizer (ρ=%.3f)", cfg.get("sam_rho", 0.05))
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)
        logger.info("Using AdamW optimizer")

    # ── Mixup ──────────────────────────────────────────────────────────────
    mixup_fn = None
    if cfg.get("mixup_alpha", 0) > 0:
        mixup_fn = Mixup(alpha=cfg["mixup_alpha"])
        logger.info("Mixup enabled (α=%.2f)", cfg["mixup_alpha"])

    # ── AMP ────────────────────────────────────────────────────────────────
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    use_amp = device.type == "cuda"

    # ── Training ───────────────────────────────────────────────────────────
    epochs = cfg.get("epochs", 30)
    warmup_epochs = cfg.get("warmup_epochs", 2)
    patience = cfg.get("patience", 10)
    save_dir = Path(cfg.get("save_dir", "./checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    best_metric = -1.0
    epochs_no_improve = 0
    use_sam = cfg.get("sam_enabled", False)

    logger.info("Starting training for %d epochs", epochs)
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        current_lr = cosine_lr_schedule(
            optimizer if not use_sam else optimizer.base_optimizer,
            epoch, epochs, warmup_epochs, lr,
        )

        running_loss = 0.0
        n_batches = 0

        for batch_idx, (clips, labels, group_ids) in enumerate(train_loader):
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Mixup (on labels and inputs)
            if mixup_fn is not None:
                clips, labels = mixup_fn(clips, labels)

            if use_sam:
                # SAM: two forward-backward passes
                # --- first pass (ascent) ---
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(pixel_values_videos=clips)
                    logits = out.logits.squeeze(-1).squeeze(-1)
                    loss = criterion(logits, labels, list(group_ids)) if isinstance(criterion, GroupAlignedBCELoss) else criterion(logits, labels)
                loss.backward()
                optimizer.first_step()
                optimizer.zero_grad()

                # --- second pass (descent) ---
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(pixel_values_videos=clips)
                    logits = out.logits.squeeze(-1).squeeze(-1)
                    loss = criterion(logits, labels, list(group_ids)) if isinstance(criterion, GroupAlignedBCELoss) else criterion(logits, labels)
                loss.backward()
                optimizer.second_step()
                optimizer.zero_grad()
            else:
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(pixel_values_videos=clips)
                    logits = out.logits.squeeze(-1).squeeze(-1)
                    loss = criterion(logits, labels, list(group_ids)) if isinstance(criterion, GroupAlignedBCELoss) else criterion(logits, labels)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    "  [%d/%d] batch %d/%d  loss=%.4f  lr=%.2e",
                    epoch + 1, epochs, batch_idx + 1, len(train_loader),
                    running_loss / n_batches, current_lr,
                )

        train_loss = running_loss / max(n_batches, 1)

        # ── Validation ─────────────────────────────────────────────────────
        val_loss, val_auc = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_auc=%.4f  lr=%.2e  (%.1fs)",
            epoch + 1, epochs, train_loss, val_loss, val_auc, current_lr, elapsed,
        )

        # ── Checkpoint ─────────────────────────────────────────────────────
        metric = val_auc  # proxy for pgmAP_ALL
        if metric > best_metric:
            best_metric = metric
            epochs_no_improve = 0
            ckpt_path = save_dir / "best_model.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": (
                    optimizer.base_optimizer.state_dict() if use_sam
                    else optimizer.state_dict()
                ),
                "val_auc": val_auc,
                "val_loss": val_loss,
                "config": cfg,
            }, ckpt_path)
            logger.info("  ✓ New best model saved (val_auc=%.4f) → %s", val_auc, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, patience)
                break

        # Also save periodic checkpoints
        if (epoch + 1) % 5 == 0:
            torch.save(
                model.state_dict(),
                save_dir / f"checkpoint_epoch{epoch + 1}.pt",
            )

    logger.info("Training complete. Best val_auc=%.4f", best_metric)


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    if yaml is None:
        raise ImportError("PyYAML required for --config. Install with: pip install pyyaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train V-JEPA 2 for collision prediction (Nexar challenge)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config (e.g. configs/nexar/champion/vjepa2_alertonly_v4.yaml)",
    )
    # Allow overriding any config key via CLI
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--val_csv", type=str, default=None)
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--sam_enabled", action="store_true", default=None)
    parser.add_argument("--alert_only", action="store_true", default=None)
    parser.add_argument("--mixup_alpha", type=float, default=None)
    parser.add_argument("--drop_path_rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load base config from YAML if provided
    if args.config:
        cfg = load_config(args.config)
        logger.info("Loaded config from %s", args.config)
    else:
        cfg = {}

    # CLI overrides
    for key in [
        "model_name", "train_csv", "val_csv", "video_dir", "save_dir",
        "epochs", "batch_size", "lr", "lora_r", "sam_enabled", "alert_only",
        "mixup_alpha", "drop_path_rate", "seed", "num_workers",
    ]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    # Defaults for required fields
    cfg.setdefault("model_name", "facebook/vjepa2-vitl-fpc64-256")
    cfg.setdefault("train_csv", "/workspace/datasets/nexar_collision/train.csv")
    cfg.setdefault("val_csv", "/workspace/datasets/nexar_collision/val.csv")
    cfg.setdefault("video_dir", "/workspace/datasets/nexar_collision/train")
    cfg.setdefault("save_dir", "./checkpoints")
    cfg.setdefault("epochs", 30)
    cfg.setdefault("batch_size", 16)
    cfg.setdefault("lr", 3e-4)

    logger.info("Config: %s", {k: v for k, v in sorted(cfg.items())})
    train(cfg)


if __name__ == "__main__":
    main()
