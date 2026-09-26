"""v9: v8 + cross-encoder score (xlm-roberta-base fine-tuned on uncertain pairs) as a stacked
feature.  Train scores are out-of-fold (same Source-1-grouped folds), NaN outside the uncertain band.

  python v9.py --work ../work_v3 --emb ../emb --out ../output_v9
"""
import argparse, json, os, time
import numpy as np, pandas as pd, lightgbm as lgb
from common import gt_pairs, fbeta_from_assign
from pipeline import assign_from_prob, write_lists, PARAMS
from stage2 import decide_expected_f
from v6 import load, decide
from v7 import prune
from v8 import pair_basics, word_odds, diff_feats, NUM_COV, DIFF

t0 = time.time()
log = lambda *a: print(time.strftime("%H:%M:%S"), f"{time.time()-t0:6.0f}s", *a, flush=True)
KEY = lambda qi, si: qi.astype(np.int64) * 10_000_000 + si.astype(np.int64)


def ce_feature(cand, ce_pairs, scores):
    s = pd.Series(scores, index=KEY(ce_pairs.q_idx.values, ce_pairs.s1_idx.values))
    return s.reindex(KEY(cand.q_idx.values, cand.s1_idx.values)).values.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--emb", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=400); ap.add_argument("--procs", type=int, default=44)
    a = ap.parse_args(); W = a.work; P = dict(PARAMS, num_threads=a.procs)
    m8 = json.load(open(f"{W}/model_meta_v8.json")); feats7 = json.load(open(f"{W}/model_meta_v7.json"))["feats"]

    s1, q, cand, F = load(W, "train", a.procs); truth = gt_pairs(W, s1, q); cand, F = prune(cand, F, m8["alpha"])
    y = (truth[cand.q_idx.values] == cand.s1_idx.values).astype(np.int8)
    ext, nc = pair_basics(cand, s1, q, a.procs)
    fold = (cand.s1_idx.values * 2654435761 % 2**32) % 2
    D = pd.DataFrame(np.zeros((len(cand), len(DIFF)), np.float32), columns=DIFF)
    for k in (0, 1):
        odds = word_odds(ext, y, fold != k); idx = np.flatnonzero(fold == k)
        D.iloc[idx] = diff_feats([ext[i] for i in idx], odds).values
    odds_all = word_odds(ext, y, np.ones(len(y), bool))
    ce = ce_feature(cand, pd.read_parquet(f"{a.emb}/ce_train.parquet", columns=["q_idx", "s1_idx"]), np.load(f"{a.emb}/ce_train_scores.npy"))
    log("train ce coverage", np.isfinite(ce).mean())
    feats = feats7 + NUM_COV + DIFF + ["ce"]
    X = np.hstack([F[feats7].to_numpy(np.float32), nc.to_numpy(np.float32), D.to_numpy(np.float32), ce[:, None]])
    del F, nc, D, ext
    oof = np.zeros(len(cand), np.float32)
    for k in (0, 1):
        tr, va = fold != k, fold == k
        m = lgb.train(P, lgb.Dataset(X[tr], y[tr], feature_name=feats), a.rounds); oof[va] = m.predict(X[va], num_threads=a.procs)
    res = [(fbeta_from_assign(assign_from_prob(cand, oof, len(q), th), truth, len(s1)), float(th), "thr") for th in np.arange(0.3, 0.95, 0.025)]
    res.append((fbeta_from_assign(decide_expected_f(cand, oof, len(q), len(s1), miss_rate=0.1), truth, len(s1)), 0.1, "expf"))
    rule = max(res); asg = decide(cand, oof, len(q), len(s1), rule)
    log("V9 OOF", rule, "(v8 was 0.97732)", {c: round(fbeta_from_assign(asg, truth, len(s1), s1.country.values == c), 5) for c in ("US", "India")})
    m = lgb.train(P, lgb.Dataset(X, y, feature_name=feats), a.rounds); m.save_model(f"{W}/model_v9.txt")
    imp = sorted(zip(m.feature_importance("gain"), feats), reverse=True)
    log("ce rank", [i for i, (g, f) in enumerate(imp) if f == "ce"])
    json.dump({"feats": feats, "rule": rule, "alpha": m8["alpha"]}, open(f"{W}/model_meta_v9.json", "w"), indent=1)
    del X, oof

    s1, q, cand, F = load(W, "test", a.procs); cand, F = prune(cand, F, m8["alpha"])
    ext, nc = pair_basics(cand, s1, q, a.procs); D = diff_feats(ext, odds_all)
    ce = ce_feature(cand, pd.read_parquet(f"{a.emb}/ce_test.parquet", columns=["q_idx", "s1_idx"]), np.load(f"{a.emb}/ce_test_scores.npy"))
    X = np.hstack([F[feats7].to_numpy(np.float32), nc.to_numpy(np.float32), D.to_numpy(np.float32), ce[:, None]]); del F, nc, D, ext
    p = m.predict(X, num_threads=a.procs); asg = decide(cand, p, len(q), len(s1), rule)
    cc = s1.country.values; n = np.bincount(asg[asg >= 0], minlength=len(s1))
    log(f"V9 TEST assigned {(asg >= 0).sum()} (v8 5771703)", {c: (round(n[cc == c].mean(), 3), round((n[cc == c] == 0).mean(), 4)) for c in sorted(set(cc))})
    os.makedirs(a.out, exist_ok=True); qid = q.entity_id.values; mk = asg >= 0
    write_lists(f"{a.out}/matching_results.tsv", "matched_entity_ids", s1.entity_id.values, asg[mk], qid[mk])
    write_lists(f"{a.out}/candidate_pairs.tsv", "candidate_entity_ids", s1.entity_id.values, cand.s1_idx.values, qid[cand.q_idx.values])
    log("written", a.out)


if __name__ == "__main__":
    main()
