# Team-beyond_baseline

Amazon ML Challenge 2026: **Business Entity Resolution**. Link Source-2/3 business records to
the deduplicated Source-1 entities (US, India and France, the last unseen in training).
The metric is macro F0.5.

| Version | Public LB | Notes |
|---|---|---|
| v1 | 0.956 | blocking + LightGBM |
| v3 | 0.921 | stage-2 group features: fooled by "sibling" businesses in test |
| **v5a** | **0.961** | + LaBSE candidates for Indic-script names (current final) |
| v6 | – | + house-number edit-type features (CV 0.9725 vs 0.9687) |

## Where to look

- `business_entity_resolution/README.md`: how to reproduce end-to-end (commands in order)
- `business_entity_resolution/src/`: the pipeline (`normalize.py` → `blocking.py` → `features.py` →
  `v5.py` / `v6.py`)
- `Documentation_template.md`: methodology write-up (for the final submission zip)
- `STATUS.md`: running log of results and decisions
- `gpu/`: Slurm job scripts used on the cluster (`cse-cpu-all` / `cse-gpu-all`)
- `make_zip.sh`: builds `<team>_submission.zip` from an output folder

The data, caches and output TSVs are **not** in the repo (competition data, and too large for GitHub).
They live on the cluster under `~/amazon_ml_2026/business_entity_resolution/`.
