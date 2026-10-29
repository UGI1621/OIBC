"""GBDT(LightGBM) 기준 모델. MLP 와 같은 피처·같은 홀드아웃으로 채점해 비교·앙상블 재료로 쓴다.

  python mlp/gbdt.py                            # 주최 측 baseline.py 설정 (600트리, lr 0.06, 잎 127), L2, 시드 0
  python mlp/gbdt.py --objective l1 --seeds 0-11  # L1 손실, 시드 12개를 한 프로세스에서 차례로 (데이터는 한 번만 준비)
  python mlp/gbdt.py --objective l1 --seeds 0-11 --final   # 학습 구간 전체로 학습해 대회 채점 구간 예측 저장
  python mlp/gbdt.py --objective l1 --seeds 0-2 --round-change   # 라운드 변화 입력 5개 추가 (라운드 17 전용)
  python mlp/gbdt.py --objective l1 --seeds 0-2 --round-change --wind-snow --from 2021-01-01 --cap-weight --neighbors   # 권장 (2026-10-29)

테스트 모드: 학습 --from ~ 2024-05 (MLP 의 학습+검증 기간), 테스트 2024-06 ~ 2025-05.
  GBDT 는 조기 종료용 세트가 따로 필요 없어 MLP 보다 두 달 더 학습한다.
  --from 기본값 2023-06-01 은 주최 측 베이스라인과 같은 시작일(베이스라인 재현용). 2021-01-01 이 벌금 -2.7%.
  --wind-snow 는 풍향 sin/cos 4개 + 눈·결빙 5개 입력 추가 (common.WS_FEATS). 벌금 -1.7%.
  --cap-weight 는 행 가중치 = 설비용량. 이용률 L1 을 kWh L1 로 바꿔 채점 단위와 맞춘다. 벌금 -2.1% (3시드 모두).
    k(11~15시 3배)·SMP 까지 곱하면 효과가 줄어(-1.3%, -0.8%) 용량만 쓴다. MLP 에 넣으면 효과 없음(+0.3%).
    가중치가 있으면 L1 잎 값 계산이 느려져 시드당 학습이 약 4배 걸린다.
저장 (mlp/outputs/): gbdt_<obj>_test_pred[_sN].parquet, gbdt_<obj>_final_pred[_sN].parquet  (site_id, ts, p 이용률)
  시드 0 은 접미사 없음. 시드에 따른 무작위는 subsample 0.8(행), colsample 0.8(열) 두 곳뿐.
  --neighbors 는 이웃 시각(1시간 전·후) 예보 8개 + 하루 요약 6개 입력 추가 (common.NB_FEATS).
    세 연도 모두 GBDT 벌금 -2.1~-2.4% (15번 비교 모두 감소), 혼합 -0.9~-1.2%.
  --round-change / --wind-snow / --from / --cap-weight / --neighbors 를 쓰면 <obj> 에 _rc / _ws / _from202101 / _cw / _nb 가 붙는다.
  'gbdt_l1_final_pred*' 같은 glob 에 다른 설정의 파일이 섞이지 않게 하기 위해서다.
"""
from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import (B, KST, NB_FEATS, OUT, RC_FEATS, WS_FEATS, add_neighbors, add_round_change, add_wind_snow,
                    build_test_frame, build_train_frame, log)

ap = argparse.ArgumentParser()
ap.add_argument("--objective", choices=["l2", "l1"], default="l2")
ap.add_argument("--final", action="store_true")
ap.add_argument("--round", type=int, default=17)
ap.add_argument("--seeds", default="0", help="'0', '0,1,2', '0-11' 형식")
ap.add_argument("--round-change", action="store_true", help="라운드 17 - 라운드 10 변화 입력 5개 추가")
ap.add_argument("--wind-snow", action="store_true", help="풍향 sin/cos 4개 + 눈·결빙 5개 입력 추가")
ap.add_argument("--from", dest="ts_from", default="2023-06-01", help="학습 시작일. 권장 2021-01-01")
ap.add_argument("--cap-weight", action="store_true", help="행 가중치 = 설비용량 (kWh 단위 L1)")
ap.add_argument("--neighbors", action="store_true", help="이웃 시각·하루 요약 예보 14개 입력 추가")
args = ap.parse_args()
seeds = [int(x) for x in args.seeds.split(",")] if "-" not in args.seeds else list(range(*[int(v) + i for i, v in enumerate(args.seeds.split("-"))]))
assert not args.round_change or args.round == 17, "--round-change 는 라운드 17 전용"
FEATS = (B.FEATS + (RC_FEATS if args.round_change else []) + (WS_FEATS if args.wind_snow else [])
         + (NB_FEATS if args.neighbors else []))

FROM = pd.Timestamp(f"{args.ts_from} 01:00", tz=KST)
TEST_FROM = None if args.final else pd.Timestamp("2024-06-01 01:00", tz=KST)
d, sites = build_train_frame(FROM, args.round)
if args.round_change:
    d = add_round_change(d, build_train_frame(FROM, 10)[0])
if args.wind_snow:
    d = add_wind_snow(d)
if args.neighbors:
    d = add_neighbors(d, "train", args.round)
ts = d.ts
train = np.ones(len(d), bool) if TEST_FROM is None else (ts < TEST_FROM).values
test = ~train
X = d[FEATS].astype("float32")
y = d.y.values
cap = d.cap.values.astype("float32")
if args.final:
    dt, _, index = build_test_frame(args.round)
    if args.round_change:
        dt = add_round_change(dt, build_test_frame(10)[0])
    if args.wind_snow:
        dt = add_wind_snow(dt)
    if args.neighbors:
        dt = add_neighbors(dt, "test", args.round)
    Xf = dt[FEATS].astype("float32")
obj = (args.objective + ("_rc" if args.round_change else "") + ("_ws" if args.wind_snow else "")
       + ("" if args.ts_from == "2023-06-01" else f"_from{args.ts_from[:7].replace('-', '')}")
       + ("_cw" if args.cap_weight else "") + ("_nb" if args.neighbors else ""))
w = cap[train] / cap[train].mean() if args.cap_weight else None   # 평균 1로 맞춘 용량 가중치
log(f"[gbdt {obj}] train {train.sum():,} | test {test.sum():,} | 600 trees | seeds {seeds} | 입력 {len(FEATS)}개")

for seed in seeds:
    params = dict(n_estimators=600, learning_rate=0.06, num_leaves=127, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1,
                  random_state=seed, objective=args.objective)   # baseline.py 와 동일 (+objective, seed)
    t0 = time.time()
    model = lgb.LGBMRegressor(**params).fit(X[train], y[train], sample_weight=w)
    sfx = ("" if args.round == 17 else f"_r{args.round}") + ("" if seed == 0 else f"_s{seed}")
    if test.any():
        p = np.clip(model.predict(X[test]), 0, None)
        yt = y[test]; ct = cap[test]
        mae = float((np.abs(p - yt) * ct).mean())
        g = pd.DataFrame({"ts": ts[test].values, "p": p * ct, "a": yt * ct}).groupby("ts").sum()
        rel = float(np.abs(g.p - g.a).sum() / g.a.sum())
        out = OUT / f"gbdt_{obj}_test_pred{sfx}.parquet"
        pd.DataFrame({"site_id": d.site_id[test].values, "ts": ts[test].values, "p": p.astype("float32")}).to_parquet(out, index=False)
        log(f"seed {seed:2d}  fit {time.time() - t0:4.0f}s  TEST MAE {mae:.3f} kWh  팀 합계 {rel * 100:.2f}%  -> {out.name}")
    else:
        p = np.clip(model.predict(Xf), 0, None)
        p = np.where(dt.elev.values > 0, p, 0.0)
        out = OUT / f"gbdt_{obj}_final_pred{sfx}.parquet"
        pd.DataFrame({"site_id": dt.site_id.values, "ts": dt.ts.values, "p": p.astype("float32")}).to_parquet(out, index=False)
        log(f"seed {seed:2d}  fit {time.time() - t0:4.0f}s  predicted {len(dt):,} rows, {(p * dt.cap.values).sum() / 1e6:.2f} GWh  -> {out.name}")
