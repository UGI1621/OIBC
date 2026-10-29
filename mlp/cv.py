"""여러 연도 검증: 같은 방법을 평가 연도 여러 개에서 벌금으로 비교한다. 한 해만 보고 고르면 그 해에만 맞는 방법을 고르기 쉽다.

  python mlp/cv.py                                       # 기본: A=rc vs B=rc,ws,long,cw, 연도 2022·2023·2024, 시드 0-2
  python mlp/cv.py --a rc,ws,long --b rc,ws,long,cw --folds 2023,2024

연도 Y = 평가 Y-06 ~ Y+1-05. 학습은 항상 평가 기간 이전 데이터만 쓴다 (미래로 과거를 맞히지 않는다).
GBDT 설정 토큰 (쉼표로 구분):
  rc   라운드 변화 5개 (common.RC_FEATS)      ws  풍향 sin/cos + 눈·결빙 9개 (common.WS_FEATS)
  long 2021-01 부터 학습 (없으면 평가 직전 1년) cw  용량 가중치
MLP: mlp/outputs/pred_big_{test|foldY}_s*.parquet 가 있으면 혼합(MLP 0.7 + GBDT 0.3)도 채점한다.
  train_big.py --fold Y --seed N 으로 만든다 (--fold 2024 는 tag 'test').
구성: snake, 그리고 mlp/outputs/composition_stn.csv 가 있으면 지점 나누기도 채점한다.
채택 규칙(권장): 모든 연도에서 같은 시드끼리 비교가 대부분 줄 때만 채택. 2022 연도는 발전소 31~80개뿐이라 참고용.
--seen-only: 학습에 있던 발전소(Y-04-01 이전에 발전량이 있는 곳)만 채점한다. 실제 대회는 모든 발전소를 학습에서 본 뒤
  예측하므로 이쪽이 대회와 더 가깝다. 검증 연도의 '처음 보는 발전소'는 MLP 편향을 해마다 크게 바꾼다
  (2023-24 +5.6%, 2024-25 −2.5%; 학습에 있던 발전소는 두 해 모두 0~+0.6%).
출력은 집계 숫자만 (벌금 백만 원, ② = 시드 평균 예측의 벌금).
"""
from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import B, DATA, KST, OUT, RC_FEATS, WS_FEATS, add_round_change, add_wind_snow, build_train_frame, log

ap = argparse.ArgumentParser()
ap.add_argument("--a", default="rc", help="기준 GBDT 설정 (기본: 10-28 상태)")
ap.add_argument("--b", default="rc,ws,long,cw", help="비교 GBDT 설정 (기본: 10-29 권장)")
ap.add_argument("--folds", default="2022,2023,2024")
ap.add_argument("--seeds", default="0-2")
ap.add_argument("--seen-only", action="store_true", help="학습에 있던 발전소만 채점 (대회와 같은 조건)")
args = ap.parse_args()
folds = [int(x) for x in args.folds.split(",")]
seeds = list(range(int(args.seeds.split("-")[0]), int(args.seeds.split("-")[-1]) + 1))


def parse(spec):
    t = set(x for x in spec.split(",") if x)
    assert t <= {"rc", "ws", "long", "cw"}, t
    return dict(feats=B.FEATS + (RC_FEATS if "rc" in t else []) + (WS_FEATS if "ws" in t else []),
                long="long" in t, cw="cw" in t, name=spec or "24개")


CFG = {"A": parse(args.a), "B": parse(args.b)}
L0 = pd.Timestamp("2021-01-01 01:00", tz=KST)
d, sites = build_train_frame(L0, 17)
d = add_wind_snow(add_round_change(d, build_train_frame(L0, 10)[0]))
smp = pd.read_csv(DATA / "train" / "smp.csv")
smp["ts"] = pd.to_datetime(smp.ts).dt.tz_convert(KST)
W = smp.set_index("ts").pipe(lambda s: s.smp * np.where(s.index.hour.isin(range(11, 16)), 3, 1))
capk = sites.set_index("site_id").capacity_kw
order = capk.sort_values(ascending=False).index
first = d.groupby("site_id").ts.min()                                   # 발전소별 첫 데이터 시각
COMPS = {"snake": pd.Series({s: ["G1", "G2", "G2", "G1"][i % 4] for i, s in enumerate(order)})}
if (OUT / "composition_stn.csv").exists():
    COMPS["지점 나누기"] = pd.read_csv(OUT / "composition_stn.csv").set_index("site_id").group
log(f"A = {CFG['A']['name']} | B = {CFG['B']['name']} | 연도 {folds} | 시드 {seeds}")

summary = []
for Y in folds:
    T1, T2 = [pd.Timestamp(f"{y}-06-01 01:00", tz=KST) for y in (Y, Y + 1)]
    te = ((d.ts >= T1) & (d.ts < T2)).values
    if args.seen_only:
        te = te & (d.site_id.map(first) < pd.Timestamp(f"{Y}-04-01 01:00", tz=KST)).values
    h = d[te].reset_index(drop=True)
    cap, y = h.cap.values, h.y.values

    def pen(p, comp="snake"):
        g = pd.DataFrame({"ts": h.ts, "grp": h.site_id.map(COMPS[comp]).values, "p": p * cap, "a": y * cap})
        t = g.groupby(["grp", "ts"])[["p", "a"]].sum().reset_index()
        return float((0.1 * t.ts.map(W).values * (t.p - t.a).abs()).sum() / 1e6)

    P = {}
    for k, c in CFG.items():
        start = L0 if c["long"] else pd.Timestamp(f"{Y - 1}-06-01 01:00", tz=KST)
        tr = ((d.ts >= start) & (d.ts < T1)).values
        wt = d.cap.values[tr] / d.cap.values[tr].mean() if c["cw"] else None
        P[k] = []
        for s in seeds:
            t0 = time.time()
            m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.06, num_leaves=127, min_child_samples=200,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                                  random_state=s, objective="l1")
            m.fit(d.loc[tr, c["feats"]].astype("float32"), d.y[tr], sample_weight=wt)
            P[k].append(np.clip(m.predict(h[c["feats"]].astype("float32")), 0, None))
            log(f"  {Y} {k} seed {s} 학습 {tr.sum():,}행 ({time.time() - t0:.0f}s)")
    pa, pb = [pen(p) for p in P["A"]], [pen(p) for p in P["B"]]
    ga, gb = np.mean(P["A"], 0), np.mean(P["B"], 0)
    row = dict(연도=f"{Y}-06~{Y + 1}-05", 발전소=h.site_id.nunique(), GBDT_A=pen(ga), GBDT_B=pen(gb),
               같은시드=", ".join(f"{100 * (b / a - 1):+.1f}%" for a, b in zip(pa, pb)))
    row["GBDT_차이%"] = 100 * (row["GBDT_B"] / row["GBDT_A"] - 1)
    # MLP 예측이 있으면 혼합
    tag = "test" if Y == 2024 else f"fold{Y}"
    mfiles = sorted(OUT.glob(f"pred_big_{tag}_s*.parquet"))
    if mfiles:
        key = h[["site_id", "ts"]]
        mp = []
        for f in mfiles:
            q = pd.read_parquet(f)
            q["ts"] = q.ts.dt.tz_localize("UTC").dt.tz_convert(KST)              # 저장 시 naive UTC
            v = key.merge(q, on=["site_id", "ts"], how="left").p.values
            assert not np.isnan(v).any(), f
            mp.append(v)
        mm = np.mean(mp, 0)
        row.update(MLP=pen(mm), 혼합_A=pen(0.7 * mm + 0.3 * ga), 혼합_B=pen(0.7 * mm + 0.3 * gb), MLP시드=len(mp))
        row["혼합_차이%"] = 100 * (row["혼합_B"] / row["혼합_A"] - 1)
        for comp in COMPS:
            if comp != "snake":
                row[f"{comp}%"] = 100 * (pen(0.7 * mm + 0.3 * gb, comp) / row["혼합_B"] - 1)
    summary.append(row)

print(f"\n[A = {CFG['A']['name']}  vs  B = {CFG['B']['name']}]  벌금 백만 원 (②), 구성 snake"
      + (", 학습에 있던 발전소만" if args.seen_only else ""))
print(pd.DataFrame(summary).set_index("연도").round(2).to_string())
