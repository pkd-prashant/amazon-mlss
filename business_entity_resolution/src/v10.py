"""v10: v9 everywhere, with transductive vocabulary adaptation for low-coverage countries.

v9's learned word-difference odds only cover ~51 % of the name words of the country that is absent
from training, which made its decisions there unreliable (v9h fell back to v7 for it).  Here the
missing word statistics are learned from the *test inputs themselves*: pairs that the
vocabulary-free v7 model is very confident about serve as pseudo-labels (p > HI = match,
p < LO = non-match) for estimating one-sided-word odds.  Words with training statistics keep
their training odds; only words unseen in training take the pseudo-label odds.  No test labels
and no external data are used.  Countries are handled as an open set, selected by coverage.

  python v10.py --work ../work_v3 --emb ../emb --out ../output_v10
"""
import argparse, json, os, time
import numpy as np, pandas as pd, lightgbm as lgb
from common import load_split
from pipeline import write_lists
from v6 import load, decide
from v7 import prune
from v8 import pair_basics, word_odds, diff_feats, NUM_COV, DIFF
from v9 import ce_feature

t0 = time.time()
log = lambda *a: print(time.strftime("%H:%M:%S"), f"{time.time()-t0:6.0f}s", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--emb", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--min_cov", type=float, default=0.8)
    ap.add_argument("--hi", type=float, default=0.97); ap.add_argument("--lo", type=float, default=0.03)
    ap.add_argument("--procs", type=int, default=44)
    a = ap.parse_args(); W = a.work
    m9 = json.load(open(f"{W}/model_meta_v9.json")); m7 = json.load(open(f"{W}/model_meta_v7.json"))
    feats7 = m7["feats"]
    odds_train = pd.read_parquet(f"{W}/v8_word_odds.parquet").log_odds.to_dict()

    s1, q, cand, F = load(W, "test", a.procs); cand, F = prune(cand, F, m9["alpha"])
    ext, nc = pair_basics(cand, s1, q, a.procs)
    # coverage per country (open set)
    qc = q.country.values[cand.q_idx.values]
    toks = pd.Series(ext).str.split().explode().dropna()
    cov_df = pd.DataFrame({"c": qc[toks.index.values], "seen": toks.isin(odds_train.keys()).values})
    cov = cov_df.groupby("c").seen.mean()
    low = set(cov[cov < a.min_cov].index)
    log("one-sided-word coverage by training vocabulary:", cov.round(3).to_dict(), "| adapting:", sorted(low))

    # pseudo-labels from the vocabulary-free v7 model
    p7 = lgb.Booster(model_file=f"{W}/model_v7.txt").predict(F[feats7].to_numpy(np.float32), num_threads=a.procs)
    is_low = np.isin(qc, list(low))
    best = np.zeros(len(cand), bool)
    best[pd.Series(p7).groupby(cand.q_idx.values).idxmax().values] = True
    pos = is_low & best & (p7 > a.hi)
    neg = is_low & (p7 < a.lo)
    lab = pos | neg
    odds_pseudo = word_odds(ext, pos.astype(np.int8), lab)
    new = {w: v for w, v in odds_pseudo.items() if w not in odds_train}
    odds = {**new, **odds_train}
    log(f"pseudo-labels: {pos.sum()} pos / {neg.sum()} neg; new word odds learned: {len(new)}")
    top = sorted(new.items(), key=lambda kv: kv[1])
    log("adapted dangerous:", [w for w, _ in top[:20]]); log("adapted harmless:", [w for w, _ in top[-20:]])
    cov2 = pd.DataFrame({"c": qc[toks.index.values], "seen": toks.isin(odds.keys()).values}).groupby("c").seen.mean()
    log("coverage after adaptation:", cov2.round(3).to_dict())

    D = diff_feats(ext, odds)
    ce = ce_feature(cand, pd.read_parquet(f"{a.emb}/ce_test.parquet", columns=["q_idx", "s1_idx"]), np.load(f"{a.emb}/ce_test_scores.npy"))
    X = np.hstack([F[feats7].to_numpy(np.float32), nc.to_numpy(np.float32), D.to_numpy(np.float32), ce[:, None]])
    del F, nc, D, ext
    p = lgb.Booster(model_file=f"{W}/model_v9.txt").predict(X, num_threads=a.procs)
    asg = decide(cand, p, len(q), len(s1), m9["rule"])
    cc = s1.country.values; n = np.bincount(asg[asg >= 0], minlength=len(s1))
    log(f"V10 TEST assigned {(asg >= 0).sum()}", {c: (round(n[cc == c].mean(), 3), round((n[cc == c] == 0).mean(), 4)) for c in sorted(set(cc))})
    os.makedirs(a.out, exist_ok=True); qid = q.entity_id.values; mk = asg >= 0
    write_lists(f"{a.out}/matching_results.tsv", "matched_entity_ids", s1.entity_id.values, asg[mk], qid[mk])
    write_lists(f"{a.out}/candidate_pairs.tsv", "candidate_entity_ids", s1.entity_id.values, cand.s1_idx.values, qid[cand.q_idx.values])
    pd.Series(new).to_frame("log_odds").to_parquet(f"{W}/v10_adapted_word_odds.parquet")
    log("written", a.out)


if __name__ == "__main__":
    main()
