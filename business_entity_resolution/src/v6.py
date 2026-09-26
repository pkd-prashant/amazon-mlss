"""v6: v5a + house-number edit-type features.

On train, top candidates whose first house number differs from the Source-1 entity's are true
matches 84 % of the time when a leading digit was dropped (1344 -> 344), but only 4-11 % of the time
when both numbers have the same length and differ by 3-50 (1817 vs 1808): the signature of
"sibling" look-alike businesses, which are even more frequent in test.  These features describe
the *kind* of number difference explicitly.

  python v6.py --work ../work_v3 --out ../output_v6
"""
import argparse, json, os, time
from multiprocessing import Pool
import numpy as np, pandas as pd, lightgbm as lgb
from rapidfuzz.distance import Levenshtein
from common import load_split, gt_pairs, fbeta_from_assign
from pipeline import assign_from_prob, write_lists, PARAMS
from stage2 import decide_expected_f

t0 = time.time()
log = lambda *a: print(time.strftime("%H:%M:%S"), f"{time.time()-t0:6.0f}s", *a, flush=True)
NUM_FEATS = ["hn_same_len", "hn_absdiff_log", "hn_reldiff", "hn_lev", "hn_prefix", "hn_suffix",
             "hn_substr", "hn_lendiff", "hn_first_digit_eq", "hn_last_digit_eq", "hn_all_nums_lev_min"]
_G = {}


def _nf(a, b, qa, sa):
    if not a or not b:
        return [-1.0] * len(NUM_FEATS)
    ia, ib = int(a[:15]), int(b[:15])
    same = len(a) == len(b)
    best = min((Levenshtein.distance(x, y) for x in qa for y in sa), default=-1)
    return [float(same), np.log1p(abs(ia - ib)), abs(ia - ib) / max(ia, ib, 1), Levenshtein.distance(a, b),
            float(a != b and (a.startswith(b) or b.startswith(a))), float(a != b and (a.endswith(b) or b.endswith(a))),
            float(a != b and (a in b or b in a)), float(len(a) - len(b)), float(a[0] == b[0]), float(a[-1] == b[-1]),
            float(best)]


def _chunk(bounds):
    lo, hi = bounds
    qi, si = _G["qi"][lo:hi], _G["si"][lo:hi]
    qn, sn = _G["qn"], _G["sn"]
    out = np.empty((hi - lo, len(NUM_FEATS)), np.float32)
    for k, (i, j) in enumerate(zip(qi, si)):
        qa, sa = qn[i], sn[j]
        out[k] = _nf(qa[0] if qa else "", sa[0] if sa else "", qa[:3], sa[:3])
    return out


def num_features(cand, s1, q, procs):
    _G.update(qi=cand.q_idx.values, si=cand.s1_idx.values,
              qn=[x.split() for x in q.a_nums.values], sn=[x.split() for x in s1.a_nums.values])
    step = 250000
    with Pool(procs) as pool:
        parts = pool.map(_chunk, [(i, min(i + step, len(cand))) for i in range(0, len(cand), step)])
    _G.clear()
    return pd.DataFrame(np.vstack(parts), columns=NUM_FEATS)


def load(W, split, procs):
    s1, q = load_split(W, split)
    d = pd.read_parquet(f"{W}/{split}_v5_cand_feat.parquet")
    cand = d[["q_idx", "s1_idx", "score", "rank"]].reset_index(drop=True)
    F = d.drop(columns=["q_idx", "s1_idx", "score", "rank"]).reset_index(drop=True)
    del d
    N = num_features(cand, s1, q, procs)
    return s1, q, cand, pd.concat([F, N], axis=1)


def decide(cand, p, n_q, n_s1, rule):
    return (assign_from_prob(cand, p, n_q, rule[1]) if rule[2] == "thr"
            else decide_expected_f(cand, p, n_q, n_s1, miss_rate=rule[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--procs", type=int, default=44)
    a = ap.parse_args()
    W = a.work
    P = dict(PARAMS, num_threads=a.procs)
    meta5 = json.load(open(f"{W}/model_meta_v5.json"))

    s1, q, cand, F = load(W, "train", a.procs)
    truth = gt_pairs(W, s1, q)
    y = (truth[cand.q_idx.values] == cand.s1_idx.values).astype(np.int8)
    feats = meta5["feats1"] + NUM_FEATS
    X = F[feats].to_numpy(np.float32); del F
    log("train", X.shape)
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
    log("V6 OOF", rule, "(v5a was", meta5["rule1"][0], ")",
        {c: round(fbeta_from_assign(asg, truth, len(s1), s1.country.values == c), 5) for c in ("US", "India")})
    m = lgb.train(P, lgb.Dataset(X, y, feature_name=feats), a.rounds)
    m.save_model(f"{W}/model_v6.txt")
    imp = sorted(zip(m.feature_importance("gain"), feats), reverse=True)
    log("new-feature importance ranks", [(i, f) for i, (g, f) in enumerate(imp) if f in NUM_FEATS])
    json.dump({"feats": feats, "rule": rule}, open(f"{W}/model_meta_v6.json", "w"), indent=1)
    del X, oof

    s1, q, cand, F = load(W, "test", a.procs)
    p = m.predict(F[feats].to_numpy(np.float32), num_threads=a.procs); del F
    asg = decide(cand, p, len(q), len(s1), rule)
    cc = s1.country.values
    n = np.bincount(asg[asg >= 0], minlength=len(s1))
    log("V6 TEST assigned", (asg >= 0).sum(), "(v5a 5672446)",
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
