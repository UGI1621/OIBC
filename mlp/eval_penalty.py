"""홀드아웃 1년(2024-06~2025-05) 예측을 대회 점수식으로 채점: 0.1 × Σ SMP × k × |팀 합계 예측 − 팀 합계 실적|.

  python mlp/eval_penalty.py

먼저 있어야 하는 파일 (mlp/outputs/):
  gbdt_l2_test_pred.parquet, gbdt_l2_test_pred_r10.parquet   주최 측 베이스라인 재현용 (gbdt.py --objective l2, --round 10)
  gbdt_l1_test_pred*.parquet                                  gbdt.py --objective l1 --seeds 0-2
  gbdt_l1_rc_test_pred*.parquet                               gbdt.py --objective l1 --seeds 0-2 --round-change (없으면 생략)
  gbdt_l1_rc_ws_from202101_test_pred*.parquet                 gbdt.py ... --round-change --wind-snow --from 2021-01-01 (없으면 생략)
  gbdt_l1_rc_ws_from202101_cw_test_pred*.parquet              위 + --cap-weight (권장, 없으면 생략)
  model_big_test_s*.pt                                        train_big.py --seed N (없으면 MLP·혼합 행은 생략)

출력: 모델별 MAE·팀 합계 오차·벌금(원), 구성별 벌금, 시각·월별 벌금 비중과 편향, GBDT 입력 중요도. 집계 숫자만 출력한다.
k 는 학습 구간 smp.csv 에 없어서 규칙(파일 표기 11~15시 = 3)으로 만든다. 주간(elev > 0) 행만 채점한다.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd, torch, lightgbm as lgb

from common import B, DATA, KST, OUT, Scaler, build_train_frame, mlp_predict  # noqa: E402

TEST_FROM = pd.Timestamp("2024-06-01 01:00", tz=KST)
d, sites = build_train_frame(TEST_FROM, 17)
d = d[d.ts < pd.Timestamp("2025-06-01 01:00", tz=KST)].reset_index(drop=True)
cap, y = d.cap.values, d.y.values
key = pd.DataFrame({"site_id": d.site_id.values, "ts": d.ts.values})  # gbdt.py 는 ts 를 naive UTC 로 저장

def gb(pattern):
    ps = [key.merge(pd.read_parquet(p), on=["site_id", "ts"], how="left").p.values for p in sorted(OUT.glob(pattern))]
    return (np.mean(ps, 0) if ps else None), len(ps)

def mlp_preds(pattern):
    out = []
    for p in sorted(OUT.glob(pattern)):
        ck = torch.load(p, weights_only=False)
        X = Scaler.from_state(ck["scaler"]).transform(d[ck["feats"]].astype("float32").values)
        out.append(mlp_predict(ck, X, d.site_id.map(ck["smap"]).values.astype("int64")))   # 에폭 평균 모델도 처리
    return np.mean(out, 0), len(out)

smp = pd.read_csv(DATA / "train" / "smp.csv"); smp["ts"] = pd.to_datetime(smp.ts).dt.tz_convert(KST)
smp["k"] = np.where(smp.ts.dt.hour.between(11, 15), 3, 1); smp["w"] = smp.smp * smp.k
W = smp.set_index("ts").w

capk = sites.set_index("site_id").capacity_kw
order = capk.sort_values(ascending=False).index
COMPS = {
    "snake": pd.Series({s: ["G1", "G2", "G2", "G1"][i % 4] for i, s in enumerate(order)}),
    "ref": sites.set_index("site_id").current_group.map({"A": "G1", "B": "G2"}),
    "one_team(하한)": pd.Series("G1", index=capk.index),
}
lat = sites.set_index("site_id").lat; COMPS["north_south"] = (lat >= lat.median()).map({True: "G1", False: "G2"})
if (OUT / "composition_stn.csv").exists():                         # compose.py 결과 (지점 나누기, 오차를 보지 않음)
    COMPS["지점 나누기"] = pd.read_csv(OUT / "composition_stn.csv").set_index("site_id").group

def penalty(p, comp="snake"):
    g = pd.DataFrame({"ts": d.ts, "grp": d.site_id.map(COMPS[comp]).values, "p": p * cap, "a": y * cap})
    t = g.groupby(["grp", "ts"])[["p", "a"]].sum().reset_index()
    t["w"] = t.ts.map(W).values
    pen = 0.1 * (t.w * (t.p - t.a).abs())
    return float(pen.sum()), t.assign(pen=pen)

def scores(p):
    mae = float((np.abs(p - y) * cap).mean())
    g = pd.DataFrame({"ts": d.ts, "p": p * cap, "a": y * cap}).groupby("ts").sum()
    return mae, float(np.abs(g.p - g.a).sum() / g.a.sum() * 100)

M = {}
r10, _ = gb("gbdt_l2_test_pred_r10.parquet"); r17, _ = gb("gbdt_l2_test_pred.parquet")
M["주최 측 베이스라인 (L2, 라운드10·17 평균)"] = (r10 + r17) / 2
M["GBDT L2 (라운드17)"] = r17
gl1, ng = gb("gbdt_l1_test_pred*.parquet"); M[f"GBDT L1 {ng}시드 평균"] = gl1
gl1_0, _ = gb("gbdt_l1_test_pred.parquet"); M["GBDT L1 시드 1개"] = gl1_0
grc, nrc = gb("gbdt_l1_rc_test_pred*.parquet")
if nrc:
    M[f"GBDT L1 + 라운드 변화 {nrc}시드 평균"] = grc
gws, nws = gb("gbdt_l1_rc_ws_from202101_test_pred*.parquet")
if nws:
    M[f"GBDT L1 + 라운드변화·풍향·눈, 2021~ {nws}시드"] = gws
gcw, ncw = gb("gbdt_l1_rc_ws_from202101_cw_test_pred*.parquet")
if ncw:
    M[f"GBDT 위 + 용량 가중치 {ncw}시드"] = gcw
mm, nm = mlp_preds("model_big_test_s*.pt")
if nm:
    M[f"큰 MLP {nm}시드 평균"] = mm
    for w in (0.5, 0.7):
        M[f"혼합 MLP {w} + GBDT {1-w:.1f}"] = w * mm + (1 - w) * gl1
    if nrc:
        M["혼합 MLP 0.7 + GBDT(라운드 변화) 0.3"] = 0.7 * mm + 0.3 * grc
    if nws:
        M["혼합 MLP 0.7 + GBDT(풍향·눈, 2021~) 0.3"] = 0.7 * mm + 0.3 * gws
    if ncw:
        M["혼합 MLP 0.7 + GBDT(권장, 용량 가중) 0.3"] = 0.7 * mm + 0.3 * gcw

base_pen, _ = penalty(M["주최 측 베이스라인 (L2, 라운드10·17 평균)"])
revenue = float((pd.DataFrame({"ts": d.ts, "a": y * cap}).groupby("ts").a.sum() * W).sum())
print(f"홀드아웃 행 {len(d):,} | 매출 항(Σ SMP×실적) {revenue/1e8:.2f}억 원\n")
print(f"{'모델':40s} {'MAE':>7s} {'팀합계%':>7s} {'벌금(snake)':>12s} {'베이스라인대비':>10s} {'벌금/매출':>8s}")
res = {}
for name, p in M.items():
    mae, rel = scores(p); pen, _ = penalty(p)
    res[name] = dict(mae=mae, rel=rel, pen=pen)
    print(f"{name:40s} {mae:7.3f} {rel:7.2f} {pen/1e6:10.1f}백만 {100*(pen/base_pen-1):+9.1f}% {100*pen/revenue:7.2f}%")

best = list(M)[-1] if nm else f"GBDT L1 {ng}시드 평균"; pb = M[best]
print(f"\n[구성별 벌금] 모델 = {best}")
lb, _ = penalty(pb, "one_team(하한)")
for c in COMPS:
    pen, _ = penalty(pb, c); print(f"  {c:16s} {pen/1e6:8.1f}백만 원  하한 대비 {100*(pen/lb-1):+5.1f}%")

_, t = penalty(pb)
t["hour"] = t.ts.dt.hour; t["mon"] = t.ts.dt.strftime("%Y-%m")
h = t.groupby("hour").agg(pen=("pen", "sum"), p=("p", "sum"), a=("a", "sum"))
h["share%"] = 100 * h.pen / h.pen.sum(); h["bias%"] = 100 * (h.p / h.a - 1)
print(f"\n[시각별] 벌금 비중과 편향(예측/실적-1), 모델 = {best}")
print(h[h.a > 0][["share%", "bias%"]].round(1).T.to_string())
mo = t.groupby("mon").agg(pen=("pen", "sum"), p=("p", "sum"), a=("a", "sum"))
mo["share%"] = 100 * mo.pen / mo.pen.sum(); mo["bias%"] = 100 * (mo.p / mo.a - 1)
print("\n[월별]"); print(mo[["share%", "bias%"]].round(1).T.to_string())

# 입력 중요도: GBDT L1 시드 0 을 학습 구간으로 다시 학습해 gain 비중을 본다
dtr, _ = build_train_frame(pd.Timestamp("2023-06-01 01:00", tz=KST), 17)
dtr = dtr[dtr.ts < TEST_FROM]
mdl = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.06, num_leaves=127, min_child_samples=200, subsample=0.8,
                        subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0, objective="l1")
mdl.fit(dtr[B.FEATS].astype("float32"), dtr.y)
imp = pd.Series(mdl.booster_.feature_importance("gain"), index=B.FEATS); imp = 100 * imp / imp.sum()
print("\n[GBDT 입력 중요도, gain 비중 %]"); print(imp.sort_values(ascending=False).round(1).to_string())
miss = d[B.FEATS].isna().mean() * 100
print("\n[홀드아웃 입력 결측률 %, 0 아닌 것]"); print(miss[miss > 0].round(2).to_string())
json.dump(res, open(OUT / "penalty_test.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"saved {OUT / 'penalty_test.json'}")
