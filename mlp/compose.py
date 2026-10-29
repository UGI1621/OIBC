"""팀 구성(G1/G2) 탐색. 홀드아웃 1년의 발전소별 예측 오차로, 벌금이 줄어드는 옮기기·맞바꾸기를 하나씩 적용한다.

  python mlp/compose.py          # -> mlp/outputs/composition_opt.csv (site_id, group)
  python mlp/predict.py --composition mlp/outputs/composition_opt.csv ...

왜: 벌금은 시각마다 팀별 합계 오차 |Σ오차| 로 매긴다. 두 팀이 같은 방향으로 틀리면 한 팀일 때(하한)와 같아지고,
  반대 방향으로 틀리면 커진다. snake(용량 순 G1,G2,G2,G1)는 용량만 고르게 나눠, 같이 틀리는 발전소끼리 한 팀에 몰릴 수 있다.
방법: snake 에서 시작해, 발전소 하나 옮기기 또는 G1·G2 하나씩 맞바꾸기 중 벌금을 가장 많이 줄이는 것을 적용. 더 줄지 않으면 멈춘다.
  두 팀 용량 20,000 kW 이상 조건을 지킨다. 2^253 가지를 다 보는 것이 아니라 최선이라는 보장은 없다.
검증 (2026-10-29): 홀수 주로 고르고 짝수 주로 확인하면 snake 대비 -0.7% (반대로도 -0.7%). 고른 기간에서는 -1.3%라
  효과의 절반은 과적합이다. 1년 전체로 고르면 홀드아웃 -1.2%로 보이지만 실제 기대는 -0.7% 안팎.
오차: 평가 모드 모델(model_big_test_s*.pt 평균 0.7 + gbdt_l1_rc_ws_from202101_cw_test_pred* 평균 0.3). 집계만 출력.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from common import DATA, KST, OUT, Scaler, build_train_frame, log, mlp_predict

MIN_KW = 20000
T1, T2 = [pd.Timestamp(s, tz=KST) for s in ("2024-06-01 01:00", "2025-06-01 01:00")]
d, sites = build_train_frame(T1, 17)
h = d[d.ts < T2].reset_index(drop=True)
del d

# 1. 홀드아웃 예측 오차 (kWh): 발전소 × 낮 시각 행렬
mp = []
for path in sorted(OUT.glob("model_big_test_s*.pt")):
    ck = torch.load(path, weights_only=False)
    X = Scaler.from_state(ck["scaler"]).transform(h[ck["feats"]].astype("float32").values)
    mp.append(mlp_predict(ck, X, h.site_id.map(ck["smap"]).values.astype("int64")))
key = pd.DataFrame({"site_id": h.site_id.values, "ts": h.ts.values})
gpaths = sorted(OUT.glob("gbdt_l1_rc_ws_from202101_cw_test_pred*.parquet"))
assert mp and gpaths, "model_big_test_s*.pt, gbdt_l1_rc_ws_from202101_cw_test_pred*.parquet 가 먼저 있어야 한다"
gp = np.mean([key.merge(pd.read_parquet(f), on=["site_id", "ts"], how="left").p.values for f in gpaths], 0)
h["e"] = (0.7 * np.mean(mp, 0) + 0.3 * gp - h.y.values) * h.cap.values
log(f"오차: MLP {len(mp)}개 + GBDT {len(gpaths)}개 혼합, 행 {len(h):,}")

smp = pd.read_csv(DATA / "train" / "smp.csv")
smp["ts"] = pd.to_datetime(smp.ts).dt.tz_convert(KST)
Wt = smp.set_index("ts").pipe(lambda s: s.smp * np.where(s.index.hour.isin(range(11, 16)), 3, 1))
E = h.pivot_table(index="site_id", columns="ts", values="e", aggfunc="sum").reindex(sites.site_id).fillna(0.0)
w = 0.1 * Wt.reindex(pd.DatetimeIndex(E.columns)).values / 1e6            # 시각별 SMP×k, 백만 원 단위
E = E.values
cap = sites.capacity_kw.values
TOT = cap.sum()

# 2. 시작 구성: predict.py 와 같은 snake
capk = sites.set_index("site_id").capacity_kw
order = capk.sort_values(ascending=False).index
snake_s = pd.Series({s: ["G1", "G2", "G2", "G1"][i % 4] for i, s in enumerate(order)}).reindex(sites.site_id)
snake = (snake_s == "G1").values                                        # True = G1


def penalty(g):
    return float((w * (np.abs(E[g].sum(0)) + np.abs(E[~g].sum(0)))).sum())


# 3. 탐색
g = snake.copy()
s1, s2 = E[g].sum(0), E[~g].sum(0)
cur = penalty(g)
n_mv = n_sw = 0
while True:
    c1 = cap[g].sum()
    d1 = np.where(g[:, None], -E, E)                                     # i 를 옮길 때 G1 합의 변화
    mv = (w * (np.abs(s1 + d1) + np.abs(s2 - d1))).sum(1)
    mv[~np.where(g, c1 - cap >= MIN_KW, (TOT - c1) - cap >= MIN_KW)] = np.inf
    i = int(np.argmin(mv))
    best, act = mv[i], ("mv", i)
    A, Bn = np.where(g)[0], np.where(~g)[0]                              # 맞바꾸기 후보: 옮기기 점수 상위 40개씩
    A, Bn = A[np.argsort(mv[A])][:40], Bn[np.argsort(mv[Bn])][:40]
    for a in A:
        dd = -E[a][None, :] + E[Bn]
        nc1 = c1 - cap[a] + cap[Bn]
        sc = (w * (np.abs(s1 + dd) + np.abs(s2 - dd))).sum(1)
        sc[(nc1 < MIN_KW) | (TOT - nc1 < MIN_KW)] = np.inf
        j = int(np.argmin(sc))
        if sc[j] < best:
            best, act = sc[j], ("sw", a, Bn[j])
    if best >= cur - 1e-6:
        break
    if act[0] == "mv":
        i = act[1]
        dlt = -E[i] if g[i] else E[i]
        g[i] = not g[i]
        n_mv += 1
    else:
        a, b = act[1], act[2]
        dlt = -E[a] + E[b]
        g[a], g[b] = False, True
        n_sw += 1
    s1, s2, cur = s1 + dlt, s2 - dlt, best

# 4. 전후 비교 (집계만)
lb = float((w * np.abs(E.sum(0))).sum())


def describe(name, gg):
    e1, e2 = E[gg].sum(0), E[~gg].sum(0)
    same = float((w * (np.sign(e1) == np.sign(e2))).sum() / w.sum())       # 두 팀이 같은 방향으로 틀린 시각 (SMP×k 가중)
    st = sites.assign(g1=gg).groupby("wa_station").g1.agg(["size", "mean"])
    st = st[st["size"] >= 2]
    mixed = float(((st["mean"] > 0) & (st["mean"] < 1)).mean())             # 발전소 2개 이상인 지점 중 양쪽 팀에 다 있는 비율
    big = capk.sort_values(ascending=False).index[:20]
    nbig = int(pd.Series(gg, index=sites.site_id)[big].sum())
    log(f"{name:6s} 벌금 {penalty(gg):6.1f}백만 (하한 대비 {100 * (penalty(gg) / lb - 1):+.1f}%) | "
        f"G1 {int(gg.sum())}개 {cap[gg].sum():,.0f} kW / G2 {int((~gg).sum())}개 {cap[~gg].sum():,.0f} kW | "
        f"두 팀이 같은 방향으로 틀린 비중 {100 * same:.1f}% | 지점이 양쪽에 나뉜 비율 {100 * mixed:.0f}% | 큰 20개 중 G1 {nbig}개")


log(f"한 팀 하한 {lb:.1f}백만 | 옮기기 {n_mv}번, 맞바꾸기 {n_sw}번, snake 와 다른 발전소 {int((g != snake).sum())}개")
describe("snake", snake)
describe("탐색", g)
assert cap[g].sum() >= MIN_KW and cap[~g].sum() >= MIN_KW
out = pd.DataFrame({"site_id": sites.site_id, "group": np.where(g, "G1", "G2")})
out.to_csv(OUT / "composition_opt.csv", index=False)
log(f"saved {OUT / 'composition_opt.csv'}")
