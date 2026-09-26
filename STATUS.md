# Status — Amazon ML Challenge 2026 (Business Entity Resolution)

_Last updated: 26 Sep, ~07:55 IST_

## TL;DR — what to do when you wake up

1. **Upload this first** (the main candidate):
   ```
   /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/output_v5a/matching_results.tsv
   ```
2. If you have a spare submission, **upload this second** (a variant of v5a with a gentle correction):
   ```
   /u/student/2025/cs25mtech14011/amazon_ml_2026/business_entity_resolution/output_v5c/matching_results.tsv
   ```
3. Tell Claude both scores. The final zip will be rebuilt with whichever is best, in one command:
   `cd ~/amazon_ml_2026 && ./make_zip.sh output_v5a <TeamName>` (or `output_v5c`, or `output` for v1).
4. Fill in the **team name** at the top of `Documentation_template.md`, and rename the zip to
   `<TeamName>_submission.zip`.

Both files pass the official validator. The zip currently contains v5a:
`~/amazon_ml_2026/TEAM_submission.zip` (223 MB).

**My expectation (honest):** v5a should be **at or above v1's 0.956**, most likely around 0.958–0.962.
Its out-of-fold score is higher than v1's (0.9687 vs 0.9656), it assigns almost the same number of test
records as v1 (5.67M vs 5.71M), and the pairs it changes compared with v1 look *cleaner* (see the
evidence below). If v5a scores below 0.956, keep v1 (`output/`) as the final.

## Leaderboard history

| Version | Uploaded | LB score | What it was |
|---|---|---|---|
| v1 | 03:30 | **0.956** | blocking + stage-1 LightGBM |
| v3 | 05:31 | 0.921 | + normalisation fixes + stage-2 group features + expected-F rule |
| v5a | 11:44 | **0.961** | blocking + **LaBSE candidates (V100)** + stage-1 LightGBM |
| v5c | 11:46 | 0.961 | v5a + label-shift correction for sibling look-alikes (tie) |
| v6 | not uploaded | – | v5a + house-number edit-type features (CV 0.9725; fewer siblings, more dropped-digit matches) |
| v7 | not uploaded | – | v6 + candidate pruning (score ≥ 0.5 × best): CV 0.9725, **9.3 cands/S1 on test (was 18.3)** |
| v8 | not uploaded | – | v7 + learned word-difference odds + number coverage: CV **0.9773** (US 0.982, India 0.970); France additions look sibling-like (vocab coverage 51 %) |
| **v8h** | _pending_ | ? | v8 where learned vocabulary covers ≥ 80 % of name words (US, India), v7 elsewhere (France) |

## What happened overnight

### 1. Root cause of v3's drop (0.956 → 0.921): sibling businesses

The test set contains many **sibling businesses**: same name, same street, a *neighbouring* house number
(1817 vs 1808, 6831 vs 6828, 1001 vs 1002), several records each, and their Source-1 entity absent.

- Stage 1 rejects them correctly.
- v3's stage-2 features counted "how many records agree with *my* number/name", so a sibling cluster
  vouched for itself and got merged: +600K mostly-wrong assignments (+10 %).

Two alternative explanations were ruled out:
- The normalisation changes are harmless (same test counts as v1).
- The "more distractors in test" theory is wrong: simulated on train, stage 2 still scored 0.971.

Measured directly, test has **34–67 % more** house-number-mismatch candidates per entity than train.

### 2. The V100 (via Slurm, your account; no action needed from you)

- A multilingual-MiniLM test on native-script names was useless (4 % top-20).
- **LaBSE name+address** is excellent for Hindi/Tamil/Bengali/… names: 90 % top-1, 94.5 % top-5,
  versus 88.9 % for our blocking's top-3.
- It is worse than our blocking on Latin-script names, so it is used for native-script records only.
- Adding 2 LaBSE candidates per native-script record raises native recall from 88.9 % to 95.6 % and
  overall recall from 96.2 % to 96.7 %.

### 3. The v5 models

| | OOF F0.5 (train) | Test assigned | Verdict |
|---|---|---|---|
| v1 | 0.9656 | 5.71M | LB 0.956 |
| **v5a**: stage 1 + LaBSE | **0.9687** (US 0.976, India 0.957) | 5.67M | ✅ recommended |
| v5b: robust stage 2 | 0.9740 | 5.88M (+3.7 %) | ❌ rejected: its extra pairs are 83 % house-number mismatches (the same profile as v3's bad additions) |
| v5c: v5a + label-shift correction | – | 5.64M | ✅ second option |

Evidence that v5a improves on v1:
- The 101K pairs v1 has and v5a drops are 55 % house-number mismatches.
- The 62K pairs v5a adds are 33 % mismatches (largely native-script records recovered by LaBSE).

## Deliverables (ready)

- `~/amazon_ml_2026/TEAM_submission.zip`: output/ (v5a), code/business_entity_resolution/ (src,
  README, requirements), Documentation_template.md
- `Documentation_template.md`: fully written (EDA, blocking, LaBSE study, features, results, the
  v3 failure analysis). Only the **team name** and **final LB score** are left to fill in.
- `README.md`: exact end-to-end reproduction commands.

## Housekeeping notes

- Heavy jobs run as Slurm batch jobs (`cse-cpu-all` / `cse-gpu-all`), because the login node killed
  long processes.
- Disk is at about 8 GB free. The v1 train caches and some v3/v4 intermediates were deleted; they are
  regenerable. v1's outputs and model are kept.
