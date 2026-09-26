"""v8: v7 + learned name-difference odds + number-set coverage features.

Error analysis of v7 (train OOF) showed:
  * rejected true matches often carry a noise number injected into the address
    ("76 C/O Himatshih ..." vs "C/O Himatshih ...") -> number-set *coverage* features;
  * merged distractors are same-address businesses that differ by one *content* word
    ("BT Green Pvt Ltd" vs "BT Iron Pvt Ltd", "... Services Garments"), whereas true matches
    differ by *filler* words (Sri, Partners, Corporation, Group ...).  The odds that a word
    appearing on only one side is harmless are learned from training pairs, out-of-fold
    (2 folds grouped by Source-1 entity) so no label information leaks into the features.

  python v8.py --work ../work_v3 --out ../output_v8
"""
import argparse, json, os, time
from collections import Counter
from multiprocessing import Pool
import numpy as np, pandas as pd, lightgbm as lgb
from common import gt_pairs, fbeta_from_assign
from normalize import skeleton_word
from pipeline import assign_from_prob, write_lists, PARAMS
from stage2 import decide_expected_f
from v6 import load, decide
from v7 import prune

t0 = time.time()
log = lambda *a: print(time.strftime("%H:%M:%S"), f"{time.time()-t0:6.0f}s", *a, flush=True)
NUM_COV = ["nc_s_in_q", "nc_q_in_s", "nc_q_extra", "nc_s_extra", "nc_s_subset", "nc_q_subset"]
DIFF = ["wd_n", "wd_min", "wd_max", "wd_sum", "wd_mean", "wd_unseen"]
_G = {}


def _toks(s):
    return {t for t in s.split() if len(t) > 1}


def _extras(qt, st):
    """words present on only one side, ignoring words whose skeleton matches the other side"""
    qs = {skeleton_word(t) for t in qt}; ss = {skeleton_word(t) for t in st}
    return [t for t in qt - st if skeleton_word(t) not in ss] + [t for t in st - qt if skeleton_word(t) not in qs]


def _chunk(bounds):
    lo, hi = bounds
    qi, si = _G["qi"][lo:hi], _G["si"][lo:hi]
    qn, sn, qw, sw = _G["qn"], _G["sn"], _G["qw"], _G["sw"]
    ext, nc = [], np.empty((hi - lo, len(NUM_COV)), np.float32)
    for k, (i, j) in enumerate(zip(qi, si)):
        ext.append(" ".join(_extras(qw[i], sw[j])))
        a, b = qn[i], sn[j]
        if a and b:
            inter = len(a & b)
            nc[k] = [inter / len(b), inter / len(a), len(a - b), len(b - a), float(b <= a), float(a <= b)]
        else:
            nc[k] = [-1, -1, len(a), len(b), -1, -1]
    return ext, nc


def pair_basics(cand, s1, q, procs):
    _G.update(qi=cand.q_idx.values, si=cand.s1_idx.values,
              qn=[set(x.split()) for x in q.a_nums.values], sn=[set(x.split()) for x in s1.a_nums.values],
              qw=[_toks(x) for x in q.n_full.values], sw=[_toks(x) for x in s1.n_full.values])
    step = 250000
    with Pool(procs) as pool:
        parts = pool.map(_chunk, [(i, min(i + step, len(cand))) for i in range(0, len(cand), step)])
    _G.clear()
    ext = [e for p in parts for e in p[0]]
    nc = pd.DataFrame(np.vstack([p[1] for p in parts]), columns=NUM_COV)
    return ext, nc


def word_odds(ext, y, mask, a=2.0):
    """log-odds that a one-sided word occurs in a true pair, relative to the base rate"""
    pos, neg = Counter(), Counter()
    for e, lab, m in zip(ext, y, mask):
        if m and e:
            (pos if lab else neg).update(e.split())
    prior = np.log((y[mask].sum() + a) / ((~y[mask].astype(bool)).sum() + a))
    return {t: np.log((pos[t] + a * np.exp(prior) / (1 + np.exp(prior))) / (neg[t] + a / (1 + np.exp(prior)))) - prior
            for t in set(pos) | set(neg) if pos[t] + neg[t] >= 5}


def diff_feats(ext, odds):
    out = np.empty((len(ext), len(DIFF)), np.float32)
    for k, e in enumerate(ext):
        ws = e.split()
        if not ws:
            out[k] = [0, 0, 0, 0, 0, 0]
            continue
        v = [odds[w] for w in ws if w in odds]
        uns = len(ws) - len(v)
        out[k] = ([len(ws), min(v), max(v), sum(v), sum(v) / len(v), uns] if v else [len(ws), 0, 0, 0, 0, uns])
    return pd.DataFrame(out, columns=DIFF)


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
    feats7 = json.load(open(f"{W}/model_meta_v7.json"))["feats"]

    s1, q, cand, F = load(W, "train", a.procs)
    truth = gt_pairs(W, s1, q)
    cand, F = prune(cand, F, a.alpha)
    y = (truth[cand.q_idx.values] == cand.s1_idx.values).astype(np.int8)
    ext, nc = pair_basics(cand, s1, q, a.procs)
    log("train basics done", len(cand))
    fold = (cand.s1_idx.values * 2654435761 % 2**32) % 2
    D = pd.DataFrame(np.zeros((len(cand), len(DIFF)), np.float32), columns=DIFF)
    for k in (0, 1):
        odds = word_odds(ext, y, fold != k)
        idx = np.flatnonzero(fold == k)
        D.iloc[idx] = diff_feats([ext[i] for i in idx], odds).values
    odds_all = word_odds(ext, y, np.ones(len(y), bool))
    top = sorted(odds_all.items(), key=lambda kv: kv[1])
    log("most dangerous one-sided words:", [w for w, _ in top[:25]])
    log("most harmless one-sided words:", [w for w, _ in top[-25:]])
    feats = feats7 + NUM_COV + DIFF
    X = np.hstack([F[feats7].to_numpy(np.float32), nc.to_numpy(np.float32), D.to_numpy(np.float32)])
    del F, nc, D, ext
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
    log("V8 OOF", rule, "(v7 was 0.97248)",
        {c: round(fbeta_from_assign(asg, truth, len(s1), s1.country.values == c), 5) for c in ("US", "India")})
    m = lgb.train(P, lgb.Dataset(X, y, feature_name=feats), a.rounds)
    m.save_model(f"{W}/model_v8.txt")
    imp = sorted(zip(m.feature_importance("gain"), feats), reverse=True)
    log("new-feature ranks", [(i, f) for i, (g, f) in enumerate(imp) if f in NUM_COV + DIFF])
    json.dump({"feats": feats, "rule": rule, "alpha": a.alpha}, open(f"{W}/model_meta_v8.json", "w"), indent=1)
    pd.Series(odds_all).to_frame("log_odds").to_parquet(f"{W}/v8_word_odds.parquet")
    del X, oof

    s1, q, cand, F = load(W, "test", a.procs)
    cand, F = prune(cand, F, a.alpha)
    ext, nc = pair_basics(cand, s1, q, a.procs)
    D = diff_feats(ext, odds_all)
    X = np.hstack([F[feats7].to_numpy(np.float32), nc.to_numpy(np.float32), D.to_numpy(np.float32)])
    del F, nc, D, ext
    p = m.predict(X, num_threads=a.procs)
    asg = decide(cand, p, len(q), len(s1), rule)
    cc = s1.country.values
    n = np.bincount(asg[asg >= 0], minlength=len(s1))
    log(f"V8 TEST cands/S1 {len(cand)/len(s1):.2f}; assigned {(asg >= 0).sum()} (v7 5696451)",
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
