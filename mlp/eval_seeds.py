"""시드 수에 따른 앙상블 이득과 GBDT·MLP 혼합 가중치를 홀드아웃 1년(2024-06~2025-05)에서 잰다.

  python mlp/eval_seeds.py [--mlp 'model_big_test_s*.pt'] [--gbdt 'gbdt_l1_test_pred*.parquet']

출력: 시드별 점수, 시드 k개 평균의 MAE(k = 1..N, 무작위 부분집합 평균), 전체 평균, 혼합 가중치별 점수.
저장: mlp/outputs/seeds_test.json
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd
import torch

from common import KST, OUT, MLP, Scaler, build_train_frame, log

ap = argparse.ArgumentParser()
ap.add_argument("--mlp", default="model_big_test_s*.pt")
ap.add_argument("--gbdt", default="gbdt_l1_test_pred*.parquet")
ap.add_argument("--subsets", type=int, default=40, help="k개 평균을 잴 때 뽑는 무작위 부분집합 수")
args = ap.parse_args()
seed_of = lambda p: int(re.search(r"_s(\d+)", p.name).group(1)) if re.search(r"_s(\d+)", p.name) else 0
mlp_paths = sorted(OUT.glob(args.mlp), key=seed_of)
gb_paths = sorted(OUT.glob(args.gbdt), key=seed_of)

d, _ = build_train_frame(pd.Timestamp("2024-06-01 01:00", tz=KST), 17)
cap = d.cap.values.astype("float32"); y = d.y.values.astype("float32"); ts = d.ts.values
key = pd.DataFrame({"site_id": d.site_id.values, "ts": ts})


tcode, _ = pd.factorize(ts)                       # 시각별 합계를 bincount 로 (groupby 보다 수십 배 빠름)
A_sum = np.bincount(tcode, y * cap)


def score(p):
    mae = float((np.abs(p - y) * cap).mean())
    P_sum = np.bincount(tcode, p * cap)
    return mae, float(np.abs(P_sum - A_sum).sum() / A_sum.sum() * 100)


def mlp_pred(path):
    ck = torch.load(path, weights_only=False)
    m = MLP(ck["nin"], ck["hidden"], ck["emb"], drop=ck.get("drop", 0.0)); m.load_state_dict(ck["state"]); m.eval()
    X = Scaler.from_state(ck["scaler"]).transform(d[ck["feats"]].astype("float32").values)
    si = torch.tensor(d.site_id.map(ck["smap"]).values.astype("int64"))
    with torch.no_grad():
        return np.clip(m(torch.tensor(X), si).numpy(), 0, None)


def gb_pred(path):
    p = key.merge(pd.read_parquet(path), on=["site_id", "ts"], how="left").p.values
    assert not np.isnan(p).any(), path
    return p


def k_curve(P, rng):
    """시드 k개를 무작위로 뽑아 평균한 MAE 의 기대값 (k = 1..N)."""
    N = len(P); out = {}
    for k in range(1, N + 1):
        if k == N:
            out[k] = score(np.mean(P, 0))[0]; continue
        vals = [score(np.mean([P[i] for i in rng.choice(N, k, replace=False)], 0))[0] for _ in range(args.subsets)]
        out[k] = float(np.mean(vals))
    return out


rng = np.random.default_rng(0)
res = {}
for name, paths, fn in (("mlp", mlp_paths, mlp_pred), ("gbdt", gb_paths, gb_pred)):
    if not paths:
        log(f"{name}: 파일 없음"); continue
    P = [fn(p) for p in paths]
    singles = [score(p) for p in P]
    maes = np.array([s[0] for s in singles])
    ens = score(np.mean(P, 0))
    curve = k_curve(P, rng)
    corr = np.corrcoef(np.array([(p - y) * cap for p in P]))
    off = corr[np.triu_indices(len(P), 1)]
    log(f"== {name}: 시드 {len(P)}개  단일 MAE {maes.mean():.3f} ± {maes.std():.3f} (최소 {maes.min():.3f} 최대 {maes.max():.3f})  "
        f"전체 평균 {ens[0]:.3f} kWh / 팀 합계 {ens[1]:.2f}%  시드 간 오차 상관 {off.mean():.3f}")
    log("   k개 평균 MAE: " + "  ".join(f"k={k}:{v:.3f}" for k, v in curve.items() if k in (1, 2, 3, 4, 6, 8, 10, 12) or k == len(P)))
    res[name] = {"seeds": [seed_of(p) for p in paths], "single_mae": maes.round(4).tolist(),
                 "single_team_rel_pct": [round(s[1], 3) for s in singles], "mean_single_mae": round(float(maes.mean()), 4),
                 "std_single_mae": round(float(maes.std()), 4), "ensemble_mae": round(ens[0], 4), "ensemble_team_rel_pct": round(ens[1], 3),
                 "k_curve": {str(k): round(v, 4) for k, v in curve.items()}, "err_corr_mean": round(float(off.mean()), 4), "_P": P}

if "mlp" in res and "gbdt" in res:
    pm = np.mean(res["mlp"]["_P"], 0); pg = np.mean(res["gbdt"]["_P"], 0)
    c = float(np.corrcoef((pm - y) * cap, (pg - y) * cap)[0, 1])
    log(f"== 혼합 (MLP 평균 vs GBDT 평균 오차 상관 {c:.3f})")
    blend = {}
    for w in (0.5, 0.6, 0.7, 0.8, 0.9):
        mae, rel = score(w * pm + (1 - w) * pg)
        blend[str(w)] = {"mae": round(mae, 4), "team_rel_pct": round(rel, 3)}
        log(f"   MLP {w:.1f} + GBDT {1 - w:.1f}  MAE {mae:.3f} kWh  팀 합계 {rel:.2f}%")
    res["blend"] = {"corr_mlp_gbdt": round(c, 4), "weights": blend}
for v in res.values():
    v.pop("_P", None)
(OUT / "seeds_test.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
log(f"saved {OUT / 'seeds_test.json'}")
