# Business Entity Resolution — reproduction guide

Pipeline:

1. Text normalisation (script-agnostic, including Indic transliteration).
2. Reverse top-K blocking with compound keys.
3. LaBSE name+address embedding neighbours for records whose name is in an Indic script.
4. LightGBM pair matcher with exclusive one-record → one-entity assignment.
5. (Optional) label-shift correction for "sibling" look-alike candidates.

No external data, lookups or APIs are used. Models:
- **LightGBM** (MIT license), trained from scratch on the provided training data.
- **LaBSE** (`sentence-transformers/LaBSE`, Apache-2.0, ~471M parameters), used frozen as a text
  encoder for native-script names only. It is downloaded once from the HuggingFace model hub; no
  business data is looked up.

## Environment

- Python 3.11, Linux. CPU steps: ~40 cores and ~160 GB RAM recommended (they run in 1–2 h). The
  embedding step needs one CUDA GPU (a V100 takes ~10 min).
- `pip install -r requirements.txt` (sentence-transformers and torch are only needed for the GPU step).
- ~25 GB free disk for cached intermediates in the work directory.

## Run end-to-end

All commands are run from `src/`. `DATA` points at the challenge `dataset/` folder.

```bash
cd src
DATA=/path/to/student_resource/dataset
WORK=../work            # cache: normalised data, candidates, features, models
EMB=../emb              # LaBSE bundle + neighbour files

# 1. normalise every record (country is an open set of labels)
python3 preprocess.py --data $DATA --work $WORK --split train
python3 preprocess.py --data $DATA --work $WORK --split test

# 2. blocking + base pair features on train (also writes the stage-1 feature list to
#    $WORK/model_meta.json) and on test
python3 pipeline.py train   --work $WORK --K 5 --Kuse 3 --df_cap 2000 --rounds 400
python3 pipeline.py predict --work $WORK --out ../output_base      # caches test candidates/features

# 3. LaBSE neighbours for native-script names (GPU)
python3 export_native.py --work $WORK --emb_dir $EMB
python3 emb_knn.py $EMB train test

# 4. final model: blocking + LaBSE candidates, stage-1 LightGBM (+ robust stage-2 variant)
#    writes ../output_v5a/{matching_results,candidate_pairs}.tsv   <- main submission
python3 v5.py --work $WORK --knn_dir $EMB --out_root .. --extra 2

# 5. (optional variant) label-shift correction for sibling look-alikes -> ../output_v5c/
python3 sibling_prior.py $WORK ../output_v5c

# 6. validate
python3 <student_resource>/utils/validate_submission.py \
    --matching ../output_v5a/matching_results.tsv --candidate ../output_v5a/candidate_pairs.tsv \
    --test-dir $DATA/test
```

## Source files (`src/`)

| File | Role |
|---|---|
| `normalize.py` | Unicode/Indic transliteration, accent stripping, legal-suffix & alias parsing, address canonicalisation (street types, ordinals, US/Indian states, French regions/departments), consonant skeletons |
| `preprocess.py` | Parallel normalisation of all source files → parquet cache |
| `blocking.py` | Reverse top-K candidate generation with compound blocking keys and an IDF-weighted sparse inverted index |
| `features.py` | Pairwise string-similarity features + candidate-competition features |
| `export_native.py`, `emb_knn.py` | Native-script bundle export and LaBSE top-20 neighbour search (GPU) |
| `emb_candidates.py` | Address-aware re-ranking of LaBSE neighbours; adds ≤ 2 candidates per native-script record, plus a LaBSE-cosine feature |
| `stage2.py` | Group-consistency features and the per-entity expected-F0.5 decision rule |
| `v5.py` | Final training (2-fold grouped CV, decision tuning) and test inference / output writing |
| `sibling_prior.py` | Label-shift correction of candidate odds for house-number-mismatch sub-populations |
| `pipeline.py` | Stage-1 base training / blocking / feature caching driver |
| `common.py` | Data loading, ground-truth alignment, exact macro-F0.5 metric |
| `dev/` | Diagnostics and experiments used during development (blocking recall, error analysis, embedding recall tests, distribution-shift checks). Not needed to reproduce the outputs |

## Outputs

- `candidate_pairs.tsv`: exactly the (Source-1, record) pairs scored by the matching model. That is the
  top-3 blocking candidates per Source-2/3 record, plus ≤ 2 LaBSE candidates per native-script record,
  grouped per Source-1 entity.
- `matching_results.tsv`: the final matches. They are a subset of the candidates, and each Source-2/3
  record is assigned to at most one Source-1 entity.
