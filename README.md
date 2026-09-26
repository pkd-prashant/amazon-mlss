# Team-beyond_baseline

Amazon ML Challenge 2026: **Business Entity Resolution**. Link Source-2/3 business records to
the deduplicated Source-1 entities (US, India and France, the last unseen in training).
The metric is macro F0.5.

| Version | Public LB | Train CV | Notes |
|---|---|---|---|
| v1 | 0.956 | 0.9656 | blocking + LightGBM |
| v3 | 0.921 | 0.9743 | stage-2 group features: fooled by "sibling" businesses in test |
| v5a | **0.961** | 0.9687 | + LaBSE candidates for Indic-script names |
| v7 | – | 0.9725 | + house-number edit-type features, candidate pruning (9.3 candidates/S1) |
| v8 | – | 0.9773 | + learned word-difference odds, number-set coverage |
| **v9h** | _pending_ | **0.9828** | + cross-encoder (xlm-roberta-base) score; France falls back to v7 (low vocabulary coverage) |
| **v10r** | _pending_ | 0.9828 | v9 everywhere; France vocabulary self-trained from confident test pairs + exact-identity restore |

## Where to look

- `business_entity_resolution/README.md`: how to reproduce end-to-end (commands in order)
- `business_entity_resolution/src/`: the pipeline (`normalize.py` → `blocking.py` → `features.py` →
  `v5.py` … `v10.py`, `cross_encoder.py`)
- `Documentation_template.md`: methodology write-up (for the final submission zip)
- `STATUS.md`: running log of results and decisions
- `gpu/`: Slurm job scripts used on the cluster (`cse-cpu-all` / `cse-gpu-all`)
- `make_zip.sh`: builds `<team>_submission.zip` from an output folder

The data, caches and output TSVs are **not** in the repo (competition data, and too large for GitHub).
They live on the cluster under `~/amazon_ml_2026/business_entity_resolution/`.
