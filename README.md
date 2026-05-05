# Orze — NeurIPS 2026 Anonymous Code Release

This is the anonymized code release accompanying the submission
*"Auto-Researching, Not Hyperparameter Tuning: Convergence Analysis of 10,000 Experiments"*.

> **Double-blind review copy.** Author-identifying strings (organization,
> usernames, upstream GitHub URLs) have been replaced with `ANON`. The
> de-anonymized repository with original licensing and provenance will be
> linked upon acceptance.

## What's here

```
orze-anon/
├── orze/                       Apache-2.0 orchestration system
├── orze-pro/                   Research-agent extension (reviewer license)
├── configs/nexar/              YAML configs for the Nexar experiments
│   ├── champion/                 deployed single-model champion recipe
│   ├── methods/                  all ablation / method configs
│   ├── portfolios/               baseline search-policy portfolios
│   └── validators/               validator / calibration configs
├── scripts/nexar/              train_vjepa2.py + TTA extractor
├── scripts/dashcam_risk/       SMAC baseline runner + analyzer
├── .env                        pre-baked reviewer license (loaded automatically)
├── .env.reviewer               backup copy of reviewer license key
├── Dockerfile                  reproducible training environment
├── LICENSE                     Apache-2.0 (upstream orze license)
└── SCRUB_NOTES.md              double-blind scrub audit
```

## orze-pro activation (no network calls required)

**Orze-Pro is the research-agent extension of Orze.** The core orchestrator
(`orze/`) is open-source (Apache-2.0). Orze-Pro (`orze-pro/`) is a
commercial extension provided here under a **reviewer license** so that all
claims in the paper can be verified without requesting access.

A pre-configured license key is included in two places:

| File | Purpose |
|------|---------|
| `.env` | Loaded automatically by `orze-pro` at import time |
| `.env.reviewer` | Backup copy; can be copied to `.env` or exported manually |

The key expires **2026-08-05** and is limited to 2 machines. To use it
manually:

```bash
# Option A: source the .env (already done by Docker and pip install)
set -a; . ./.env; set +a

# Option B: export directly
export ORZE_PRO_KEY="ORZE-PRO-eyJjdXN0b21lciI6Im5pcHMtcmV2aWV3IiwidGllciI6InBybyIsIm1heF9tYWNoaW5lcyI6MiwiZXhwaXJlcyI6IjIwMjYtMDgtMDUifQ.qubHdBgwRIQkUgWNx52fiTGJs4EL3A5xtm8tVJQIYC5eCtrC4-1h3f7E7NrWw7WuP6vdmHBd5_PPempRXTwFDQ"
```

The activation endpoint is deliberately pointed at an invalid host
(`disabled.for.review.invalid`) so `license.py` falls through to its
air-gapped offline-pending branch and **no network traffic leaves the
reviewer's machine**.

Verify from a shell:

```bash
set -a; . ./.env; set +a
python -c "from orze_pro.license import license_info; print(license_info())"
# Licensed to nips-review (pro), expires 2026-08-05
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

## De-anonymization

This repository is fully anonymized for double-blind review. Author
identities, organization names, and upstream repository URLs are replaced
with `ANON`. Full de-anonymization (including conflict-of-interest
disclosures) will be provided upon acceptance.

## Licensing

- `orze/` — Apache-2.0 (unchanged from upstream).
- `orze-pro/` — upstream license is proprietary; a **reviewer license key**
  (expires 2026-08-05, 2 machines) is pre-configured in `.env` and
  `.env.reviewer` so all experiments can be reproduced without requesting
  access. Original license restored upon de-anonymization.
