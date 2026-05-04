# Orze — NeurIPS 2026 Anonymous Code Release

This is the anonymized code release accompanying the submission
*"Self-Evolving Search Spaces: The η² Boundary Between Auto-Tuning
and Auto-Research"*.

> **Double-blind review copy.** Author-identifying strings (organization,
> usernames, upstream GitHub URLs) have been replaced with `ANON`. The
> de-anonymized repository with original licensing and provenance will be
> linked upon acceptance.

## What's here

```
orze-anon/
├── orze/                       Apache-2.0 orchestration system
├── orze-pro/                   Intelligence layer (agents, procedures, SOPs)
├── configs/nexar/              YAML configs for the Nexar experiments
│   ├── champion/                 deployed single-model champion recipe
│   ├── methods/                  all ablation / method configs
│   ├── portfolios/               baseline search-policy portfolios
│   └── validators/               validator / calibration configs
├── scripts/nexar/              train_vjepa2.py + TTA extractor
├── scripts/dashcam_risk/       SMAC baseline runner + analyzer
├── .env                        pre-baked orze-pro license (review copy)
├── Dockerfile                  reproducible training environment
├── LICENSE                     Apache-2.0 (upstream orze license)
└── SCRUB_NOTES.md              double-blind scrub audit
```

## orze-pro activation (no network calls required)

The `.env` at the repository root contains a valid orze-pro license key
baked in for review. Its signature is verified against the Ed25519 public
key embedded in `orze-pro/src/orze_pro/license.py`. The activation endpoint
is deliberately pointed at an invalid host (`disabled.for.review.invalid`)
so `license.py` falls through to its air-gapped offline-pending branch and
**no network traffic leaves the reviewer's machine**.

Verify from a shell:

```bash
set -a; . ./.env; set +a
python -c "from orze_pro.license import license_info; print(license_info())"
# Licensed to admin (pro), expires 2027-12-31
```

## Companion artifacts (public Hugging Face)

Checkpoints, a 200-run sample of experiment logs, and all precomputed
analysis JSONs are hosted at a public Hugging Face repository (URL in the
paper's *Artifacts* section) — no authentication required.

- `best_model.pt` — champion V-JEPA 2 checkpoint (`alertonly_v4`, 0.910 mAP)
- `val_logloss_analysis.json` — per-run raw metrics
- `bulletproof_proxy.json` — recomputed proxy analysis with field docstrings

## Quick start (reproduce the champion)

```bash
# 1. Build the container (bakes the review license in too)
docker build -t orze-anon .

# 2. Install orze + orze-pro editable
pip install -e orze/ -e orze-pro/

# 3. Train the champion recipe
python scripts/nexar/train_vjepa2.py \
    --config configs/nexar/champion/vjepa2_alertonly_v4.yaml

# 4. 12-TTA inference
python scripts/nexar/extract_tta_dense_end.py \
    --ckpt /path/to/best_model.pt \
    --n_tta 12 \
    --mean_pool False   # champion uses attentive probe, NOT mean pooling
```

## Running the orchestration system

```bash
orze run orze.yaml.example
```

See `orze/README.md` and `orze-pro/README.md` for full documentation.

## Licensing

- `orze/` — Apache-2.0 (unchanged from upstream).
- `orze-pro/` — upstream license is proprietary; a review-only notice is
  included in `orze-pro/LICENSE`. Original license restored upon
  de-anonymization.
