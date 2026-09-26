"""v7: v6 + candidate pruning.

A record keeps a lower-ranked blocking candidate only if its blocking score is at least
ALPHA x the score of the record's best candidate (LaBSE candidates are always kept).  On train
this halves the candidate set (14.7 -> 7.1 per Source-1 entity) at a recall cost of 0.01 %.
Candidate-competition features are recomputed on the pruned set and the model is retrained,
so candidate_pairs.tsv is exactly what the model scores.

  python v7.py --work ../work_v3 --out ../output_v7 --alpha 0.5
"""
import argparse, json, os, time
import numpy as np, pandas as pd, lightgbm as lgb
from common import gt_pairs, fbeta_from_assign
from features import add_context
from pipeline import assign_from_prob, write_lists, PARAMS
from stage2 import decide_expected_f
from v6 import load, decide, NUM_FEATS

t0 = time.time()
log = lambda *a: print(time.strftime("%H:%M:%S"), f"{time.time()-t0:6.0f}s", *a, flush=True)


def prune(cand, F, alpha):
    top = cand[cand["rank"] == 0].set_index("q_idx").score
    ratio = cand.score.values / np.maximum(top.reindex(cand.q_idx.values).fillna(0).values, 1e-9)
    keep = (cand["rank"].values == 0) | (F.is_emb.values == 1) | (ratio >= alpha)
    c = cand[keep].reset_index(drop=True)
    F2 = F[keep].reset_index(drop=True)
    add_context(F2, c)
    return c, F2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--procs", type=int, default=44)
    a = ap.parse_args()
    W = a.work
    P = dict(PARAMS, num_threads=a.procs)
    feats = json.load(open(f"{W}/model_meta_v6.json"))["feats"]

    s1, q, cand, F = load(W, "train", a.procs)
    truth = gt_pairs(W, s1, q)
    n0 = len(cand)
    cand, F = prune(cand, F, a.alpha)
    y = (truth[cand.q_idx.values] == cand.s1_idx.values).astype(np.int8)
    log(f"train pruned {n0} -> {len(cand)} pairs, cands/S1 {len(cand)/len(s1):.2f}, recall ceiling {y.sum()/(truth>=0).sum():.4f}")
    X = F[feats].to_numpy(np.float32); del F
    fold = (cand.s1_idx.values * 2654435761 % 2**32) % 2
    oof = np.zeros(len(cand), np.float32)
    for k in (0, 1):
        tr, va = fold != k, fold == k
        m = lgb.train(P, lgb.Dataset(X[tr], y[tr], feature_name=feats), a.rounds)
        oof[va] = m.predict(X[va], num_threads=a.procs)
    res = [(fbeta_from_assign(assign_from_prob(cand, oof, len(q), th), truth, len(s1)), float(th), "thr")
           for th in np.arange(0.3, 0.95, 0.025)]
    res.append((fbeta_from_assign(decide_expected_f(cand, oof, len(q), len(s1), miss_rate=0.1), truth, len(s1)), 0.1, "expf"))
    rule = max(res)
    asg = decide(cand, oof, len(q), len(s1), rule)
    log("V7 OOF", rule, "(v6 was 0.97254)",
        {c: round(fbeta_from_assign(asg, truth, len(s1), s1.country.values == c), 5) for c in ("US", "India")})
    m = lgb.train(P, lgb.Dataset(X, y, feature_name=feats), a.rounds)
    m.save_model(f"{W}/model_v7.txt")
    json.dump({"feats": feats, "rule": rule, "alpha": a.alpha}, open(f"{W}/model_meta_v7.json", "w"), indent=1)
    del X, oof

    s1, q, cand, F = load(W, "test", a.procs)
    n0 = len(cand)
    cand, F = prune(cand, F, a.alpha)
    p = m.predict(F[feats].to_numpy(np.float32), num_threads=a.procs); del F
    asg = decide(cand, p, len(q), len(s1), rule)
    cc = s1.country.values
    n = np.bincount(asg[asg >= 0], minlength=len(s1))
    log(f"V7 TEST pruned {n0} -> {len(cand)} pairs, cands/S1 {len(cand)/len(s1):.2f}; assigned {(asg >= 0).sum()} (v6 5700304)",
        {c: (round(n[cc == c].mean(), 3), round((n[cc == c] == 0).mean(), 4)) for c in sorted(set(cc))})
    os.makedirs(a.out, exist_ok=True)
    qid = q.entity_id.values
    mk = asg >= 0
    write_lists(f"{a.out}/matching_results.tsv", "matched_entity_ids", s1.entity_id.values, asg[mk], qid[mk])
    write_lists(f"{a.out}/candidate_pairs.tsv", "candidate_entity_ids", s1.entity_id.values,
                cand.s1_idx.values, qid[cand.q_idx.values])
    log("written", a.out)


if __name__ == "__main__":
    main()
