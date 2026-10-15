"""검증 모델(model_val_*.pt)의 구간별 성능 집계 → mlp/outputs/segments.json

  python mlp/eval_segments.py [--model mlp/outputs/model_val_r17_s0.pt]

구간: 시각(hour), 월(month), 발전소 용량(capacity bucket), 예보 운량(wa_cloud bucket).
지표: 발전소·시간당 MAE(kWh), 편향(예측합/실적합 - 1), 팀 합계 상대오차, SMP×k 가중 벌금 비중.
검증 구간은 2024-06-01 ~ 2025-05-31, 주간(elev>0) 행만. 벌금은 253개소를 한 팀으로 본 값.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import DATA, KST, OUT, MLP, Scaler, build_train_frame, log

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=None)
args = ap.parse_args()
path = Path(args.model) if args.model else OUT / "model_val_r17_s0.pt"
ck = torch.load(path, weights_only=False)
model = MLP(ck["nin"], ck["hidden"], ck["emb"], drop=ck.get("drop", 0.0))
model.load_state_dict(ck["state"])
model.eval()
scaler = Scaler.from_state(ck["scaler"])

VAL_FROM = pd.Timestamp("2024-06-01 01:00", tz=KST)
d, sites = build_train_frame(VAL_FROM, ck["round"])
X = scaler.transform(d[ck["feats"]].astype("float32").values)
si = torch.tensor(d.site_id.map(ck["smap"]).values.astype("int64"))
with torch.no_grad():
    p = np.clip(model(torch.tensor(X), si).numpy(), 0, None)
d["pk"] = p * d.cap.values          # 예측 kWh
d["ak"] = d.energy.values            # 실적 kWh
d["err"] = d.pk - d.ak
d["aerr"] = d.err.abs()
log(f"val rows {len(d):,}  MAE {d.aerr.mean():.2f} kWh")

# SMP × k (학습 구간 smp.csv 에는 k 가 없으므로 11~15시 = 3 규칙으로 생성)
smp = pd.read_csv(DATA / "train" / "smp.csv")
smp["ts"] = pd.to_datetime(smp.ts)
smp = smp.set_index("ts").smp
hour = d.ts.dt.hour
d["hour"] = hour.values
d["month"] = d.ts.dt.strftime("%Y-%m").values
d["k"] = np.where((hour >= 11) & (hour <= 15), 3, 1)
d["smp"] = d.ts.map(smp).values

# 팀 합계(시각별)
t = d.groupby("ts").agg(P=("pk", "sum"), A=("ak", "sum"), n=("site_id", "size")).reset_index()
t["hour"] = t.ts.dt.hour
t["month"] = t.ts.dt.strftime("%Y-%m")
t["k"] = np.where((t.hour >= 11) & (t.hour <= 15), 3, 1)
t["smp"] = t.ts.map(smp).values
t["pen"] = 0.1 * t.smp * t.k * (t.P - t.A).abs()
t["rev"] = t.smp * t.A
total_pen = t.pen.sum()
log(f"team-sum rel error {abs(t.P - t.A).sum() / t.A.sum() * 100:.2f}%  penalty {total_pen / 1e8:.3f}억원  "
    f"revenue {t.rev.sum() / 1e8:.2f}억원  penalty/revenue {total_pen / t.rev.sum() * 100:.2f}%")


def seg(df_site, df_team, key, order=None):
    g1 = df_site.groupby(key).agg(mae=("aerr", "mean"), pk=("pk", "sum"), ak=("ak", "sum"), n=("aerr", "size"))
    g2 = df_team.groupby(key).agg(P=("P", "sum"), A=("A", "sum"), abs_team=("P", lambda s: 0), pen=("pen", "sum"))
    g2["abs_team"] = df_team.assign(e=(df_team.P - df_team.A).abs()).groupby(key).e.sum()
    g = g1.join(g2, how="left")
    g["bias_pct"] = (g.pk / g.ak - 1) * 100
    g["team_rel_pct"] = g.abs_team / g.A * 100
    g["pen_share_pct"] = g.pen / total_pen * 100
    g["energy_share_pct"] = g.ak / df_site.ak.sum() * 100
    if order is not None:
        g = g.reindex(order)
    g = g.reset_index().rename(columns={key: "key"})
    g["key"] = g.key.astype(str)
    cols = ["key", "n", "mae", "bias_pct", "team_rel_pct", "pen_share_pct", "energy_share_pct"]
    return [{c: (round(float(r[c]), 3) if c != "key" else r[c]) for c in cols} for _, r in g.iterrows()]


out = {"model": path.name, "val_from": "2024-06-01", "val_to": "2025-05-31",
       "overall": {"mae": round(float(d.aerr.mean()), 3),
                   "bias_pct": round(float((d.pk.sum() / d.ak.sum() - 1) * 100), 3),
                   "team_rel_pct": round(float(abs(t.P - t.A).sum() / t.A.sum() * 100), 3),
                   "penalty_krw": round(float(total_pen)), "revenue_krw": round(float(t.rev.sum()))}}

out["hour"] = seg(d, t, "hour", order=list(range(5, 21)))
out["month"] = seg(d, t, "month", order=sorted(d.month.unique()))

# 용량 구간 (발전소 단위 속성이므로 팀 합계 지표는 생략, 발전소·시간 지표만)
cb = pd.cut(d.cap, [0, 100, 300, 700, 2000], labels=["~100kW", "100~300kW", "300~700kW", "700kW~"])
d["capb"] = cb.astype(str)
gc = d.groupby("capb").agg(mae=("aerr", "mean"), pk=("pk", "sum"), ak=("ak", "sum"), n=("aerr", "size"),
                           abs_sum=("aerr", "sum"), sites=("site_id", "nunique"))
gc["bias_pct"] = (gc.pk / gc.ak - 1) * 100
gc["abs_err_share_pct"] = gc.abs_sum / d.aerr.sum() * 100
gc["energy_share_pct"] = gc.ak / d.ak.sum() * 100
gc["util_mae_pct"] = d.assign(u=np.abs(d.pk - d.ak) / d.cap).groupby("capb").u.mean() * 100
gc = gc.reindex(["~100kW", "100~300kW", "300~700kW", "700kW~"]).reset_index()
out["capacity"] = [{"key": r.capb, "sites": int(r.sites), "n": int(r.n), "mae": round(float(r.mae), 3),
                    "util_mae_pct": round(float(r.util_mae_pct), 3), "bias_pct": round(float(r.bias_pct), 3),
                    "abs_err_share_pct": round(float(r.abs_err_share_pct), 3),
                    "energy_share_pct": round(float(r.energy_share_pct), 3)} for _, r in gc.iterrows()]

# 예보 운량 구간 (제공사 A, 결측은 별도)
cl = pd.cut(d.wa_cloud, [-1, 20, 50, 80, 100], labels=["맑음 0~20", "구름조금 20~50", "구름많음 50~80", "흐림 80~100"])
d["cloudb"] = cl.astype(str).where(d.wa_cloud.notna(), "예보 결측")
gl = d.groupby("cloudb").agg(mae=("aerr", "mean"), pk=("pk", "sum"), ak=("ak", "sum"), n=("aerr", "size"), abs_sum=("aerr", "sum"))
gl["bias_pct"] = (gl.pk / gl.ak - 1) * 100
gl["abs_err_share_pct"] = gl.abs_sum / d.aerr.sum() * 100
gl["util_mae_pct"] = d.assign(u=np.abs(d.pk - d.ak) / d.cap).groupby("cloudb").u.mean() * 100
gl["row_share_pct"] = gl.n / len(d) * 100
gl = gl.reindex(["맑음 0~20", "구름조금 20~50", "구름많음 50~80", "흐림 80~100", "예보 결측"]).dropna(subset=["n"]).reset_index()
out["cloud"] = [{"key": r.cloudb, "n": int(r.n), "mae": round(float(r.mae), 3), "util_mae_pct": round(float(r.util_mae_pct), 3),
                 "bias_pct": round(float(r.bias_pct), 3), "abs_err_share_pct": round(float(r.abs_err_share_pct), 3),
                 "row_share_pct": round(float(r.row_share_pct), 3)} for _, r in gl.iterrows()]

(OUT / "segments.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
log(f"saved {OUT / 'segments.json'}")
for k in ("hour", "month", "capacity", "cloud"):
    print(f"\n== {k}")
    print(pd.DataFrame(out[k]).to_string(index=False))
