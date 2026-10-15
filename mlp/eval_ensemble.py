"""테스트 모드 모델 여러 개를 홀드아웃 1년(2024-06~2025-05)에서 각각·평균으로 채점.

  python mlp/eval_ensemble.py --models mlp/outputs/model_big_test_s0.pt,mlp/outputs/model_big_test_s1.pt,...
  (--models 생략 시 mlp/outputs/model_big_test_s*.pt 전부)

출력: 시드별 MAE·팀 합계 상대오차, 시드 편차, 평균(앙상블) 점수, 시드 간 오차 상관. mlp/outputs/ensemble_test.json 저장.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import KST, OUT, MLP, Scaler, build_train_frame, log

ap = argparse.ArgumentParser()
ap.add_argument("--models", default=None)
args = ap.parse_args()
paths = [Path(p) for p in args.models.split(",")] if args.models else sorted(OUT.glob("model_big_test_s*.pt"))
assert paths, "모델 없음"
cks = [torch.load(p, weights_only=False) for p in paths]
assert len({ck["round"] for ck in cks}) == 1

TEST_FROM = pd.Timestamp("2024-06-01 01:00", tz=KST)
d, _ = build_train_frame(TEST_FROM, cks[0]["round"])
cap = d.cap.values.astype("float32")
y = d.y.values.astype("float32")
ts = d.ts.values


def score(p):
    mae = float((np.abs(p - y) * cap).mean())
    g = pd.DataFrame({"ts": ts, "p": p * cap, "a": y * cap}).groupby("ts").sum()
    return mae, float(np.abs(g.p - g.a).sum() / g.a.sum() * 100)


preds, rows = [], []
for path, ck in zip(paths, cks):
    m = MLP(ck["nin"], ck["hidden"], ck["emb"], drop=ck.get("drop", 0.0))
    m.load_state_dict(ck["state"])
    m.eval()
    X = Scaler.from_state(ck["scaler"]).transform(d[ck["feats"]].astype("float32").values)
    si = torch.tensor(d.site_id.map(ck["smap"]).values.astype("int64"))
    with torch.no_grad():
        p = np.clip(m(torch.tensor(X), si).numpy(), 0, None)
    preds.append(p)
    mae, rel = score(p)
    rows.append({"model": path.name, "seed": ck.get("seed"), "best_epoch": ck.get("best_epoch"), "mae": round(mae, 3), "team_rel_pct": round(rel, 3)})
    log(f"{path.name:28s} best epoch {ck.get('best_epoch'):2}  MAE {mae:6.3f} kWh  팀 합계 {rel:5.2f}%")

maes = np.array([r["mae"] for r in rows])
ens_mae, ens_rel = score(np.mean(preds, axis=0))
errs = np.array([(p - y) * cap for p in preds])
corr = np.corrcoef(errs) if len(preds) > 1 else np.array([[1.0]])
log(f"시드 {len(rows)}개  MAE 평균 {maes.mean():.3f} ± {maes.std(ddof=0):.3f} (최소 {maes.min():.3f}, 최대 {maes.max():.3f})")
log(f"앙상블(예측 평균)  MAE {ens_mae:.3f} kWh  팀 합계 {ens_rel:.2f}%")
if len(preds) > 1:
    off = corr[np.triu_indices(len(preds), 1)]
    log(f"시드 간 오차 상관  평균 {off.mean():.3f} (범위 {off.min():.3f}~{off.max():.3f})")
summary = {"models": rows, "seed_mae_mean": float(maes.mean()), "seed_mae_std": float(maes.std(ddof=0)),
           "ensemble_mae": round(ens_mae, 3), "ensemble_team_rel_pct": round(ens_rel, 3),
           "error_corr": corr.round(4).tolist()}
(OUT / "ensemble_test.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
log(f"saved {OUT / 'ensemble_test.json'}")
