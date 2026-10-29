"""저장된 모델로 대회 채점 구간(2025-06-01 ~ 2026-06-01, 8,760시각) 예측.

  python mlp/predict.py                                    # 기본: mlp/outputs/model_big_final_s*.pt 전부 평균
  python mlp/predict.py --models a.pt,b.pt,c.pt            # 지정한 모델들의 예측 평균 (시드 앙상블)
  python mlp/predict.py --model a.pt                       # 모델 하나
  옵션 --composition snake|ref|<csv 경로>   (csv 는 site_id, group 열. 권장: mlp/outputs/composition_stn.csv, compose.py 로 생성)

출력 (mlp/outputs/):
  composition.csv            발전소 253개소의 G1/G2 배정
  forecast.csv               그룹별·시각별 예측 kWh (17,520행)
  per_site_forecast.parquet  발전소별 예측 (구성 최적화용)
구성 snake = 용량 큰 순서로 G1,G2,G2,G1 반복 배정. ref = sites.csv 의 current_group(A->G1, B->G2).
  csv = compose.py 의 지점 나누기 (같은 예보 지점 발전소를 두 팀에 번갈아). 혼합 벌금 2024-25 −0.5%, 2023-24 −0.8%.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import OUT, Scaler, build_test_frame, log, mlp_predict

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=None, help="모델 파일 하나")
ap.add_argument("--models", default=None, help="쉼표로 구분한 모델 파일 여러 개. 예측(이용률)을 평균한다")
ap.add_argument("--composition", default="snake", help="snake | ref | 구성 csv 경로 (site_id, group)")
ap.add_argument("--gbdt", default=None, help="GBDT 최종 예측 parquet glob (예: 'gbdt_l1_final_pred*.parquet'). 여러 개면 평균")
ap.add_argument("--w-mlp", type=float, default=0.7, help="--gbdt 가 있을 때 MLP 가중치. GBDT 는 1 - w")
args = ap.parse_args()

if args.models:
    paths = [Path(p) for p in args.models.split(",")]
elif args.model:
    paths = [Path(args.model)]
else:
    paths = sorted(OUT.glob("model_big_final_s*.pt")) or [OUT / "model_big_final_s0.pt"]   # 메인 = 큰 신경망
assert paths and all(p.exists() for p in paths), paths

cks = [torch.load(p, weights_only=False) for p in paths]
rounds = {ck["round"] for ck in cks}
assert len(rounds) == 1, f"모델들의 예보 라운드가 다름: {rounds}"
log(f"models: {[p.name for p in paths]} (round {rounds.pop()})")

d, sites, index = build_test_frame(cks[0]["round"])
preds = []
for p, ck in zip(paths, cks):
    X = Scaler.from_state(ck["scaler"]).transform(d[ck["feats"]].astype("float32").values)
    sidx = d.site_id.map(ck["smap"]).values.astype("int64")
    preds.append(mlp_predict(ck, X, sidx))       # 에폭 평균 모델이면 저장된 에폭 전부의 예측 평균
    n_ep = len(ck.get("states") or [ck["state"]])
    log(f"  {p.name}: {'epoch avg ' + str(n_ep) if n_ep > 1 else 'best epoch ' + str(ck.get('best_epoch'))}, "
        f"mean util {preds[-1].mean():.4f}")
p = np.mean(preds, axis=0)                       # 시드 앙상블 = 이용률 평균
if args.gbdt:
    gpaths = sorted(OUT.glob(args.gbdt))
    assert gpaths, f"GBDT 예측 없음: {args.gbdt}"
    key = pd.DataFrame({"site_id": d.site_id.values, "ts": d.ts.values})
    gps = []
    for gp in gpaths:
        v = key.merge(pd.read_parquet(gp), on=["site_id", "ts"], how="left").p.values
        assert not np.isnan(v).any(), gp
        gps.append(v)
    pg = np.mean(gps, axis=0)
    log(f"blend: MLP {args.w_mlp:.2f} + GBDT {1 - args.w_mlp:.2f} ({len(gpaths)} files, mean util {pg.mean():.4f})")
    p = args.w_mlp * p + (1 - args.w_mlp) * pg
p = np.where(d.elev.values > 0, p, 0.0)          # 야간은 0으로 고정
d["kwh"] = p * d.cap.values
log(f"predicted {len(d):,} rows ({d.site_id.nunique()} sites x {len(index):,} h), total {d.kwh.sum() / 1e6:.2f} GWh")

# 구성
cap = sites.set_index("site_id").capacity_kw
if args.composition == "ref":
    grp = sites.set_index("site_id").current_group.map({"A": "G1", "B": "G2"})
elif args.composition.endswith(".csv"):
    grp = pd.read_csv(args.composition).set_index("site_id").group.reindex(sites.site_id)
    assert grp.notna().all(), "구성 파일에 없는 발전소가 있다"
else:
    assert args.composition == "snake", args.composition
    order = cap.sort_values(ascending=False).index
    pattern = ["G1", "G2", "G2", "G1"]
    grp = pd.Series({s: pattern[i % 4] for i, s in enumerate(order)}).reindex(sites.site_id)
grp.index.name = "site_id"
c1, c2 = cap[grp == "G1"].sum(), cap[grp == "G2"].sum()
assert c1 >= 20000 and c2 >= 20000, f"capacity constraint violated: G1 {c1}, G2 {c2}"
log(f"composition={args.composition}: G1 {int((grp == 'G1').sum())}개소 {c1:,.0f} kW / "
    f"G2 {int((grp == 'G2').sum())}개소 {c2:,.0f} kW")

d["group"] = d.site_id.map(grp)
f = (d.groupby(["group", "ts"], as_index=False).kwh.sum()
       .rename(columns={"kwh": "forecast_kwh"})
       .sort_values(["group", "ts"]).reset_index(drop=True))

# 제출 형식 검증
assert len(grp) == 253 and grp.isin(["G1", "G2"]).all() and grp.index.is_unique
assert len(f) == 2 * len(index) == 17520, len(f)
assert f.forecast_kwh.ge(0).all() and np.isfinite(f.forecast_kwh).all()
assert pd.DatetimeIndex(f.ts).tz is not None
assert (f.groupby("group").ts.nunique() == len(index)).all()

grp.rename("group").reset_index().to_csv(OUT / "composition.csv", index=False)
f.to_csv(OUT / "forecast.csv", index=False)
d[["site_id", "ts", "kwh"]].to_parquet(OUT / "per_site_forecast.parquet", index=False, compression="zstd")
log(f"saved {OUT / 'composition.csv'} ({len(grp)}행), {OUT / 'forecast.csv'} ({len(f):,}행), per_site_forecast.parquet")
