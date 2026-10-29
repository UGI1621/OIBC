"""팀 구성(G1/G2) 만들기. 기본은 '지점 나누기': 같은 A사 예보 지점(wa_station)을 쓰는 발전소들을 두 팀에 번갈아 나눈다.

  python mlp/compose.py              # -> mlp/outputs/composition_stn.csv (site_id, group). 예측 오차를 보지 않는다
  python mlp/compose.py --search     # 참고용: 홀드아웃 오차로 snake 를 고치는 탐색 -> composition_search.csv (권장하지 않음)
  python mlp/predict.py --composition mlp/outputs/composition_stn.csv ...

왜: 벌금은 시각마다 팀별 합계 오차 |Σ오차| 로 매긴다. 두 팀이 같은 방향으로 틀리면 한 팀일 때(하한)와 같아지고,
  반대 방향으로 틀리면 커진다. 같은 예보 지점 발전소들은 같은 예보를 받아 같이 틀리므로, 두 팀에 나눠 넣으면
  예보가 틀릴 때 두 팀이 함께 틀린다. snake(용량 순 G1,G2,G2,G1)는 용량만 고르게 나눈다.
지점 나누기 규칙: 지점마다 발전소를 용량 순으로 G1/G2 번갈아 배정. 발전소 수가 홀수인 지점이 나오면 다음 지점의
  시작 팀을 바꿔 두 팀 크기를 맞춘다. 두 팀 용량 20,000 kW 이상이 안 되면 큰 팀의 가장 작은 발전소를 옮긴다.
검증 (2026-10-29, 혼합 모델, snake 대비): 2024-25 −0.5%, 2023-24 −0.8%.
  --search(옮기기·맞바꾸기 탐색)는 2024-25 로 고르면 그 해 −1.2%(고르지 않은 주로 확인하면 −0.7%)였지만,
  2023-24 에서는 +0.2%라 다른 해로 옮겨지지 않았다. 어느 발전소끼리 같이 틀리는지가 해마다 달라서로 본다.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import OUT, load_sites, log

MIN_KW = 20000
ap = argparse.ArgumentParser()
ap.add_argument("--search", action="store_true", help="참고용: 홀드아웃 오차로 탐색 (다른 해로 안 옮겨짐)")
args = ap.parse_args()
sites = load_sites()
cap = sites.capacity_kw.values
capk = sites.set_index("site_id").capacity_kw


def ok(g):
    return cap[g].sum() >= MIN_KW and cap[~g].sum() >= MIN_KW


def station_split():
    g, flip = np.zeros(len(cap), bool), False                            # True = G1
    for _, idx in sites.groupby("wa_station").groups.items():
        idx = np.array(sorted(idx, key=lambda i: -cap[i]))
        for j, i in enumerate(idx):
            g[i] = (j % 2 == 0) != flip
        flip = not flip if len(idx) % 2 else flip
    while not ok(g):                                                     # 용량 조건 맞추기
        g1_short = cap[g].sum() < MIN_KW
        cand = np.where(~g if g1_short else g)[0]
        g[cand[np.argmin(cap[cand])]] = g1_short
    return g


def snake():
    order = capk.sort_values(ascending=False).index
    s = pd.Series({k: ["G1", "G2", "G2", "G1"][i % 4] for i, k in enumerate(order)}).reindex(sites.site_id)
    return (s == "G1").values


def search(start):
    """홀드아웃 1년 오차로 벌금이 줄어드는 옮기기/맞바꾸기를 하나씩 적용. 평가 모드 예측이 있어야 한다."""
    import torch
    from common import DATA, KST, Scaler, build_train_frame, mlp_predict
    T1, T2 = [pd.Timestamp(s, tz=KST) for s in ("2024-06-01 01:00", "2025-06-01 01:00")]
    d, _ = build_train_frame(T1, 17)
    h = d[d.ts < T2].reset_index(drop=True)
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
    smp = pd.read_csv(DATA / "train" / "smp.csv")
    smp["ts"] = pd.to_datetime(smp.ts).dt.tz_convert(KST)
    Wt = smp.set_index("ts").pipe(lambda s: s.smp * np.where(s.index.hour.isin(range(11, 16)), 3, 1))
    E = h.pivot_table(index="site_id", columns="ts", values="e", aggfunc="sum").reindex(sites.site_id).fillna(0.0)
    w = 0.1 * Wt.reindex(pd.DatetimeIndex(E.columns)).values / 1e6
    E, TOT, g = E.values, cap.sum(), start.copy()
    s1, s2 = E[g].sum(0), E[~g].sum(0)
    cur = float((w * (np.abs(s1) + np.abs(s2))).sum())
    while True:
        c1 = cap[g].sum()
        d1 = np.where(g[:, None], -E, E)
        mv = (w * (np.abs(s1 + d1) + np.abs(s2 - d1))).sum(1)
        mv[~np.where(g, c1 - cap >= MIN_KW, (TOT - c1) - cap >= MIN_KW)] = np.inf
        i = int(np.argmin(mv))
        best, act = mv[i], ("mv", i)
        A, Bn = np.where(g)[0], np.where(~g)[0]
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
            return g
        if act[0] == "mv":
            i = act[1]
            dlt = -E[i] if g[i] else E[i]
            g[i] = not g[i]
        else:
            a, b = act[1], act[2]
            dlt = -E[a] + E[b]
            g[a], g[b] = False, True
        s1, s2, cur = s1 + dlt, s2 - dlt, best


def describe(name, g):
    st = sites.assign(g1=g).groupby("wa_station").g1.agg(["size", "mean"])
    st = st[st["size"] >= 2]
    mixed = float(((st["mean"] > 0) & (st["mean"] < 1)).mean())
    log(f"{name:6s} G1 {int(g.sum())}개 {cap[g].sum():,.0f} kW / G2 {int((~g).sum())}개 {cap[~g].sum():,.0f} kW | "
        f"발전소 2개 이상인 지점 중 두 팀에 나뉜 비율 {100 * mixed:.0f}%")


base = snake()
g = search(base) if args.search else station_split()
assert ok(g)
describe("snake", base)
describe("search" if args.search else "지점", g)
log(f"snake 와 다른 발전소 {int((g != base).sum())}개")
path = OUT / ("composition_search.csv" if args.search else "composition_stn.csv")
pd.DataFrame({"site_id": sites.site_id, "group": np.where(g, "G1", "G2")}).to_csv(path, index=False)
log(f"saved {path}")
