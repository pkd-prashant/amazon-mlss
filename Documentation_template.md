# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** Vaishnavi Goriga, Prashant Kumar Dubey, Neha Samanvitha Valiveti, Aayushi Raj  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

A CPU pipeline of rule-based, script-agnostic normalisation, *reverse* top-K blocking over compound
blocking keys, and a LightGBM pair matcher with an exclusive "one record → at most one Source-1 entity"
assignment. Each Source-2/3 record is linked to its single best Source-1 candidate, which keeps the
candidate set at ≈ 4.7 records per Source-1 entity per retained rank (the theoretical minimum for this
data is ≈ 4.7 records per entity in total). For records whose name is in an Indic script, a frozen
LaBSE encoder (Apache-2.0) adds up to 2 cross-script candidates per record. No external data, lookups or
APIs are used.

The final model reaches **macro F0.5 = 0.969 (out-of-fold, train)** with a blocking recall ceiling of
96.7 % at 14.7 candidates per Source-1 entity.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from exploratory analysis of the training data (2.21 M Source-1, 5.03 M Source-2,
5.29 M Source-3 records; US and India):

- **Every Source-2/3 record matches at most one Source-1 entity.** The ground truth has 7.64 M pairs and
  7.64 M distinct matched IDs. 73–75 % of Source-2/3 records match some entity; the rest are distractors.
  This turns matching into an assignment problem and allows *reverse* blocking (see §3).
- **Singletons are rare (5.6 %)**, and an entity has on average 3.46 matches (up to 11). Recall matters
  more than the precision-heavy metric suggests.
- **Names are weak identifiers and addresses are strong.** 31 % of Source-1 names are duplicated after
  normalisation (the vocabulary of business-name words is small and reused), but only 3.5 % of addresses
  are. Some true matches have completely unrelated names (DBA / "formerly known as" / random brand
  names) and share only the address.
- **Name noise:** legal-suffix changes (Pvt/Private, Ltd/Limited, Inc, LLC, SARL, SAS, EURL…), added filler
  words (Services, Center, Partners, Group), word-order swaps, OCR-like digit substitutions (`5ecure`,
  `C0mpany`), aliases (`X dba Y`, `formerly known as`, `a/k/a`), website forms (`lawrenceventures.com`),
  acronyms (`TCI` = *Talava Certified Ishares*), and **native Indic scripts** (≈ 12 % of Indian names in
  Devanagari, Gujarati, Tamil, Telugu, Kannada, Malayalam, Bengali, Gurmukhi, Oriya).
- **Address noise:** component re-ordering, street-type abbreviations (St/Street/**Saint**, Rd, Ave, Dr),
  full vs. abbreviated state names, states in native script (`महाराष्ट्र`), perturbed house numbers
  (1344 → 344, 4514 → 4514D, 709 → 00709), spelled-out ordinals (FIFTEENTH), `null` placeholders,
  and ≈ 3 % of records with **no address**.
- **Look-alike distractors:** many unmatched records are near-copies of a real Source-1 entity with a
  mutated house number (`287 Wiggles Ct` vs `2870 Wiggles Ct`). These are "sibling" businesses whose
  own Source-1 entity is absent, and they are the main source of false merges.
- **Test-set shift:** the test set adds France (15 % of Source-1), which is absent from training, and has
  **5.75 Source-2/3 records per Source-1 entity versus 4.68 in train**. That is consistent with a higher
  share of orphan distractor records whose Source-1 entity is absent. §5 discusses how this affected us.

### 2.2 Solution Strategy

**Approach Type:** Blocking + classifier (two-stage gradient boosting) with exclusive assignment  
**Core Innovation:** Reverse, per-record blocking over *compound* keys built on consonant skeletons
(robust to typos and cross-script transliteration), combined with the one-to-one record-assignment
structure, which gives candidate sets close to the minimum possible size.

Pipeline: `normalize → reverse top-K blocking (+ LaBSE neighbours for native-script names) → pair
features → LightGBM → best-candidate assignment + decision rule → output`.

---

## 3. Candidate Generation (Blocking)

### Normalisation (`normalize.py`)

- **Indic transliteration without external libraries:** a table is built from Unicode character names
  (e.g. "DEVANAGARI LETTER KA", "TAMIL VOWEL SIGN AA", "SIGN VIRAMA"). Consonants carry an inherent "a",
  vowel signs replace it, virama removes it, anusvara maps to "n", and a word-final schwa is deleted. The
  nine major Indic scripts share one layout, so a single routine handles all of them
  (`राम मार्केटिंग प्राइवेट लिमिटेड → ram marketing praivet limited`).
- **Consonant skeleton:** a phonetic key per word: lowercase, `ph→f`, `w→v`, `c/q→k`, `x→ks`, `g/z→j`,
  drop `h` and vowels, collapse repeats, drop a trailing `s`. Examples: `riyal/real → rl`,
  `enarji/energy → nrj`, `praivet/private → prvt`. This absorbs typos and transliteration variants.
- **Names:** accent stripping, `&`/`+` → "and", leet-digit repair, alias splitting (dba / fka / aka /
  "doing business as" / "formerly known as"), website-domain extraction, removal of legal forms and
  filler words (including their transliterated skeletons), giving *full*, *core*, *alias* and *space-less*
  views.
- **Addresses:** split on commas; drop whole components that are US states, Indian states (Latin or
  native script, matched by skeleton) or French regions/departments; canonicalise street types and
  directions (English and French: rue/R, boulevard/BD, avenue/AV, chemin, impasse…); map ordinals
  (FIFTEENTH → 15th); remove stop tokens (no, flat, unit, near, `null`, `N°`…); extract numbers with
  leading zeros stripped.
- **Country-agnostic:** the country label is only used to partition blocking. It is never a model feature,
  and the set of countries is read from the data (open set). France is handled like any other label.

### Blocking keys (`blocking.py`)

Single tokens are poor keys here because the name vocabulary is heavily reused, so most keys are compound:

| Key | Meaning |
|---|---|
| `N:` | full sorted name skeleton |
| `b:` | pairs of name-token skeletons |
| `i:` | acronym / initials |
| `p:` | first 5 chars of the space-less name (web domains) |
| `k:` | single name-token skeleton |
| `a:` / `#:` | single address word / number |
| `h:` | house number × address word |
| `w:` | consecutive address-word bigram |
| `x:` | name skeleton × address word |

- Keys whose Source-1 document frequency exceeds 2,000 are dropped (stop keys), which bounds the work
  per query like a classic block-size cap.
- Score = IDF-weighted cosine over key sets, computed as a sparse matrix product (Source-2/3 chunk ×
  Source-1ᵀ) within each country label, parallelised over 36 processes.
- **Reverse direction:** for each Source-2/3 record, keep its top-3 Source-1 candidates. A Source-1
  entity's candidate list is the set of records that ranked it in their top-3.

**Blocking keys used:** compound name-skeleton, acronym, house-number × street, address bigram and
name × address keys, in an IDF-weighted inverted index.

### Cross-script embedding candidates (`export_native.py`, `emb_knn.py`, `emb_candidates.py`)

Blocking recall is weakest when the name is in an Indic script and the address is short: blocking's
top-3 recall is 88.9 % for native-script records versus 96.8 % for the rest. We tested multilingual
sentence encoders on 20K native-script training records, measuring top-k recall of the true Source-1
entity among all Indian Source-1 records:

| Encoder (frozen) | Input | top-1 | top-5 | top-20 |
|---|---|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 | name | 0.8 % | 2.2 % | 4.2 % |
| paraphrase-multilingual-MiniLM-L12-v2 | name + address | 19.2 % | 26.2 % | 32.2 % |
| LaBSE | name | 16.2 % | 34.4 % | 60.7 % |
| **LaBSE** | **name + address** | **90.1 %** | **94.5 %** | **96.7 %** |

LaBSE was trained for cross-lingual sentence alignment, and it maps `रियल इंडस्ट्रीज` and
`Real Industries` close together. On Latin-script records it is worse than our blocking (US top-3
93.6 % vs ~98 %), so it is used **only for native-script names**:

- LaBSE name+address top-20 neighbours are computed on a V100 (~10 min for train + test).
- They are re-ranked with house-number and address-word Jaccard.
- The best 2 neighbours not already among the record's candidates are appended.
- The LaBSE cosine is added as a model feature.

Effect on train:
- native-script recall 88.9 % → 95.6 %
- overall pair recall 96.2 % → 96.7 %
- cost: +0.7 candidates per Source-1 entity

**Candidate pairs generated:**

| Split | Retained rank | Pair recall (train) | Candidates per Source-1 entity |
|---|---|---|---|
| train | top-1 | 94.3 % | 4.68 |
| train | top-2 | 95.6 % | 9.35 |
| train | top-3 (used) | 96.2 % | 14.0 |
| train | top-3 + 2 LaBSE (final) | **96.7 %** | 14.7 |
| test | top-3 + 2 LaBSE (final) | – | 18.3 (31.6 M pairs) |

For comparison, all-pairs comparison within country would be about 10⁶ candidates per entity. The
reduction ratio exceeds 99.998 %.

**How we ensured true matches were not lost:**
- Several independent key families mean a record is still retrieved when either its name or its address
  is badly corrupted.
- Skeletons tolerate typos and transliteration, and acronym keys catch initials.
- Recall was measured directly on the full training set after every change. The first single-token
  version reached 86.6 % recall@1; compound keys raised it to 94.3 %.

---

## 4. Matching Model

**Features used (42 stage-1 features, all language-agnostic):**
- **Name features:** Levenshtein ratio, token-set / token-sort / partial ratio (core and full names), best
  token-set ratio over alias parts, Jaro-Winkler on space-less names (web domains), skeleton ratio, token
  Jaccard, first-token equality, acronym match, name lengths, alias and website flags.
- **Address features:** ratio / token-set / token-sort / partial ratio, address-word token-set ratio,
  Jaccard and one-sided coverage, first-number equality, suffix-number match (1344 vs 344), whether the
  first number appears among the other record's numbers, number-set Jaccard, number counts, empty-address
  flags.
- **Other features (competition among candidates):** blocking score, rank, margin to the record's best
  candidate, gap between the record's 1st and 2nd candidates, number of candidates per record and per
  Source-1 entity.
- **Stage-2 group-consistency features:** for each Source-1 entity's candidate group, the probability
  mass and count of other records that agree with this record's house number or name versus the Source-1
  entity's own house number or name, plus rank and margin within the group. These separate the true
  entity's records from a look-alike sibling's records.

**Model type:** LightGBM binary classifier (MIT license), 400 rounds, 127 leaves, trained from scratch on
the provided data, with 44 features (the 42 above plus LaBSE cosine and an embedding-candidate flag).
LaBSE (Apache-2.0, ~471M parameters) is used frozen, only as a text encoder for candidate generation
and one feature.

**Threshold selection method:**
- Validation is 2-fold out-of-fold, grouped by Source-1 entity, and scored with an exact re-implementation
  of the macro F0.5 (singletons included).
- Each record is assigned to its highest-probability candidate, which enforces the one-entity-per-record
  structure. It is accepted if the probability exceeds a threshold tuned for macro F0.5.
- An alternative per-entity expected-F0.5 rule chooses the top-m records per entity (including m = 0)
  that maximise expected F0.5.

---

## 5. Results & Error Analysis

| Version | Pipeline | OOF macro F0.5 (train) | Test assignments | Public LB |
|---|---|---|---|---|
| v1 | blocking + stage-1 LightGBM, global threshold | 0.9656 | 5.71 M | **0.956** |
| v3 | + normalisation fixes + stage-2 group features + expected-F rule | 0.9743 | 6.26 M | 0.921 |
| **v5a (final)** | blocking + LaBSE candidates + stage-1 LightGBM, expected-F rule | **0.9687** (US 0.976, India 0.957) | 5.67 M | **0.961** |
| v5c (variant) | v5a + label-shift correction for sibling look-alikes | – | 5.64 M | 0.961 |

- **F_0.5 Score (macro), public leaderboard:** **0.961** for the final submission (v5a), up from 0.956
  for the first version.
- v5c (label-shift correction) tied at 0.961, so the simpler v5a is submitted as final.

- **Common false positives (wrong merges):** look-alike sibling businesses at the same street with a
  mutated house number (`287` vs `2870 Wiggles Ct`, `432` vs `4321 Yellow Rose Rd`); different businesses
  sharing one address (co-working / commercial complexes in India); records with empty addresses whose
  name is shared by several Source-1 entities.
- **Common false negatives (missed matches):** native-script names combined with very short addresses
  (`रियल इंडस्ट्रीज … | H.NO 27, NORTH WEST DELHI`); records with no address and a generic name; true
  matches with large house-number perturbations (`3513` vs `3502`) that look like sibling distractors.
- **Loss breakdown (stage 1, out-of-fold):** 77 % of the F0.5 loss comes from entities with missed
  matches only, 23 % from entities with at least one false merge, and singletons contribute 4.5 %
  (singleton accuracy 97.3 %).

### Lesson: a validation gain that did not transfer (stage 2)

A second-stage model with *group-consistency* features raised out-of-fold F0.5 from 0.9656 to 0.9743.
It still lowered the leaderboard score from 0.956 to 0.921. We traced the cause as follows:

- On test, stage 2 added +10 % assignments (+600K records), versus +1 % on train. That asymmetry was the
  signature.
- Most of the added pairs were **sibling businesses**: same name, same street, a *neighbouring* house
  number (1817 vs 1808, 6831 vs 6828, 1001 vs 1002), and several mutually consistent records each.
  Stage 1 rejected them (p ≈ 0.1–0.4). Stage-2 features that count support for a record's *own* house
  number or name let the sibling cluster vouch for itself (p ≈ 0.7–0.99).
- Measured directly, top-ranked candidates whose house number disagrees with the Source-1 entity are
  **34 % more frequent per entity in test** (same name: 0.87 vs 0.65) and **67 % more frequent**
  (different name: 1.52 vs 0.91). The estimated share of true matches in these groups falls from
  41.7 % to 31.2 % and from 21.4 % to 12.8 % respectively.
- We ruled out two alternatives:
  - The normalisation changes are neutral: v3 normalisation with stage 1 only gives 5.71 M test
    assignments, the same as v1.
  - A generic distractor-density shift, simulated by dropping 19 % of Source-1 entities in train
    (5.77 records per entity, as in test), did not reproduce the failure: stage 2 still scored 0.9714.

What we changed as a result:
- Stage 2 is not used in the final model. A "robust" stage 2 without self-support features still
  added +3.7 % test assignments, 83 % of which had a house-number mismatch, so it was rejected too.
- Every candidate submission must now pass a **test-side sanity gate**: its number of assignments must
  stay close to stage 1's, and its added pairs must not be dominated by house-number mismatches.
- v5c applies a label-shift correction: the odds of candidates in the two mismatch groups are scaled by
  0.63 and 0.54, the ratio of estimated test to train priors.

---

## 6. Conclusion

The main lever in this challenge is **blocking**, because recall is the dominant loss (77 % of the
out-of-fold F0.5 loss came from missed matches):

- Compound, skeleton-based keys raised recall@1 from 86.6 % to 94.3 %.
- The reverse, per-record formulation keeps the candidate set near the minimum possible size.
- A frozen cross-lingual encoder closes most of the remaining gap for Indic-script names.

The main lesson is about **distribution shift**. The test set contains many more "sibling" look-alike
businesses than train. Features that let a cluster of records vouch for itself raised validation F0.5
but failed on test. Checking test-side statistics (assignment counts and the house-number-mismatch
profile of changes) before trusting a validation gain would have caught this, and it is now part of the
pipeline.
