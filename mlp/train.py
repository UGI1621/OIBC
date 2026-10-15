"""MLP 학습.

  python mlp/train.py --mode val     # 2024-06~2025-05 를 검증으로 떼어 기존 실험(V3 12.31)과 비교
  python mlp/train.py --mode final   # 학습 구간 전체로 학습해 제출용 모델 저장

기본 설정: hidden (128,64), 임베딩 8, L1 손실, 배치 4096, AdamW lr 2e-3, 코사인 스케줄,
시간순으로 뗀 마지막 두 달에서 조기 종료. 모델·전처리 통계는 mlp/outputs/model_*.pt 에 저장.
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import B, KST, OUT, MLP, Scaler, build_train_frame, log, site_index

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["val", "final"], default="val")
ap.add_argument("--round", type=int, default=17, help="사용할 예보 발행 라운드 (10 또는 17)")
ap.add_argument("--hidden", default="128,64")
ap.add_argument("--emb", type=int, default=8)
ap.add_argument("--drop", type=float, default=0.0)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--bs", type=int, default=4096)
ap.add_argument("--patience", type=int, default=6)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--from", dest="ts_from", default="2023-06-01", help="학습 시작일 (baseline 과 같이 최근 2년)")
ap.add_argument("--no-early-stop", action="store_true", help="조기 종료 없이 --epochs 만큼 전부 학습 (학습 곡선 확인용). 최적 에폭 가중치는 그대로 복원")
ap.add_argument("--tag", default="", help="저장 파일 이름 뒤에 붙일 꼬리표")
ap.add_argument("--seg-history", action="store_true", help="(val 모드) 매 에폭 검증 세트를 시각·월·운량·용량 구간별로 집계해 history 에 기록")
args = ap.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
hidden = tuple(int(x) for x in args.hidden.split(","))

FROM = pd.Timestamp(f"{args.ts_from} 01:00", tz=KST)
if args.mode == "val":
    INNER = pd.Timestamp("2024-04-01 01:00", tz=KST)   # 조기 종료용 두 달
    MID = pd.Timestamp("2024-06-01 01:00", tz=KST)     # 이후 1년이 검증
else:
    INNER = pd.Timestamp("2025-04-01 01:00", tz=KST)
    MID = None

d, sites = build_train_frame(FROM, args.round)
ts = d.ts
fit = (ts < INNER).values
inner = ((ts >= INNER) & (ts < MID)).values if MID is not None else (ts >= INNER).values
va = (ts >= MID).values if MID is not None else np.zeros(len(d), bool)
log(f"mode={args.mode} round={args.round} hidden={hidden} emb={args.emb} seed={args.seed}")
log(f"rows {len(d):,} | fit {fit.sum():,} | early-stop {inner.sum():,} | val {va.sum():,}")

F = d[B.FEATS].astype("float32").values
y = d.y.values.astype("float32")
cap = d.cap.values.astype("float32")
scaler = Scaler().fit(F[fit])          # 통계는 fit 행에서만
X = scaler.transform(F)
known = set(d.site_id[fit].unique())
smap = site_index(sites, known)
sidx = d.site_id.map(smap).values.astype("int64")


def tensors(mask):
    return torch.tensor(X[mask]), torch.tensor(y[mask]), torch.tensor(sidx[mask])


Xt, yt, st = tensors(fit)
Xi, yi, si = tensors(inner)
ci = cap[inner]
yi_np = yi.numpy()
if args.mode == "val":                       # 매 에폭 검증 세트도 재서 학습 곡선을 남긴다
    Xv, yv, sv = tensors(va)
    cv = cap[va]
    yv_np = yv.numpy()
    ts_va = ts[va].values


def val_metrics(m):
    with torch.no_grad():
        pv = np.clip(m(Xv, sv).numpy(), 0, None)
    mae = float((np.abs(pv - yv_np) * cv).mean())
    g = pd.DataFrame({"ts": ts_va, "p": pv * cv, "a": yv_np * cv}).groupby("ts").sum()
    return mae, float(np.abs(g.p - g.a).sum() / g.a.sum()), pv


SEG = {}
if args.mode == "val" and args.seg_history:
    dv = d[va]
    hr = dv.ts.dt.hour.values
    SEG["hour"] = (np.where((hr >= 6) & (hr <= 20), hr - 6, -1), [str(h) for h in range(6, 21)])
    months = sorted(dv.ts.dt.strftime("%Y-%m").unique())
    SEG["month"] = (dv.ts.dt.strftime("%Y-%m").map({m: i for i, m in enumerate(months)}).values, months)
    cl = dv.wa_cloud.values
    cc = np.full(len(dv), 4)                      # NaN 은 비교가 False 라 4(결측) 로 남는다
    cc[cl <= 20] = 0; cc[(cl > 20) & (cl <= 50)] = 1; cc[(cl > 50) & (cl <= 80)] = 2; cc[cl > 80] = 3
    SEG["cloud"] = (cc, ["맑음 0~20", "구름조금 20~50", "구름많음 50~80", "흐림 80~100", "예보 결측"])
    cp = dv.cap.values
    SEG["capacity"] = (np.where(cp <= 100, 0, np.where(cp <= 300, 1, np.where(cp <= 700, 2, 3))),
                       ["~100kW", "100~300kW", "300~700kW", "700kW~"])


def seg_metrics(pv):
    """구간별 MAE(kWh), 편향(%), 이용률 MAE(%). bincount 로 빠르게 집계."""
    ae = np.abs(pv - yv_np) * cv; pk = pv * cv; ak = yv_np * cv; u = ae / cv
    out = {}
    for name, (codes, keys) in SEG.items():
        m = codes >= 0; c = codes[m]; k = len(keys)
        cnt = np.bincount(c, minlength=k).astype(float); cnt[cnt == 0] = np.nan
        mae = np.bincount(c, ae[m], k) / cnt
        bias = (np.bincount(c, pk[m], k) / np.bincount(c, ak[m], k) - 1) * 100
        um = np.bincount(c, u[m], k) / cnt * 100
        tolist = lambda a, nd: [None if np.isnan(v) else round(float(v), nd) for v in a]
        out[name] = {"mae": tolist(mae, 3), "bias_pct": tolist(bias, 2), "util_mae_pct": tolist(um, 3)}
    return out


history = []

model = MLP(X.shape[1], hidden, args.emb, drop=args.drop)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
crit = nn.L1Loss()
best = (1e9, None, -1)
bad = 0
n = len(Xt)
for ep in range(args.epochs):
    model.train()
    perm = torch.randperm(n)
    tot = 0.0
    for i in range(0, n, args.bs):
        b = perm[i:i + args.bs]
        sb = st[b].clone()
        sb[torch.rand(len(sb)) < 0.05] = 0            # 5%는 '모르는 발전소' 토큰으로 바꿔 0번도 학습
        opt.zero_grad()
        loss = crit(model(Xt[b], sb), yt[b])
        loss.backward()
        opt.step()
        tot += loss.item() * len(b)
    sched.step()
    model.eval()
    with torch.no_grad():
        pi = np.clip(model(Xi, si).numpy(), 0, None)
    mi = float((np.abs(pi - yi_np) * ci).mean())
    rec = {"epoch": ep + 1, "train_l1": tot / n, "inner_mae": mi, "lr": opt.param_groups[0]["lr"]}
    if args.mode == "val":
        rec["val_mae"], rec["val_team_rel"], pv_ep = val_metrics(model)
        if SEG:
            rec["seg"] = seg_metrics(pv_ep)
        log(f"epoch {ep + 1:2d}  train L1 {tot / n:.4f}  early-stop MAE {mi:.3f} kWh"
            f"  val MAE {rec['val_mae']:.3f} kWh  val 합계 {rec['val_team_rel'] * 100:.2f}%")
    else:
        log(f"epoch {ep + 1:2d}  train L1 {tot / n:.4f}  early-stop MAE {mi:.3f} kWh")
    history.append(rec)
    if mi < best[0] - 1e-4:
        best = (mi, copy.deepcopy(model.state_dict()), ep)
        bad = 0
    else:
        bad += 1
        if bad >= args.patience and not args.no_early_stop:
            log(f"early stop at epoch {ep + 1}")
            break
model.load_state_dict(best[1])
model.eval()
log(f"best epoch {best[2] + 1}  early-stop MAE {best[0]:.3f} kWh")

if args.mode == "val":
    mae, rel, _ = val_metrics(model)
    log(f"VAL 2024-06~2025-05  MAE {mae:.2f} kWh/site-hour  합계 상대오차 {rel * 100:.2f}%"
        f"  (기존 실험 V3 12.31 / GBDT 12.26)")

suffix = f"{args.mode}_r{args.round}_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
hist_path = OUT / f"history_{suffix}.json"
hist_path.write_text(json.dumps({"args": vars(args), "best_epoch": best[2] + 1,
                                 "seg_keys": {k: v[1] for k, v in SEG.items()}, "history": history},
                                ensure_ascii=False, indent=1), encoding="utf-8")
log(f"saved {hist_path}")

path = OUT / f"model_{suffix}.pt"
torch.save({"state": model.state_dict(), "hidden": hidden, "emb": args.emb, "drop": args.drop,
            "nin": X.shape[1], "scaler": scaler.state(), "smap": smap, "feats": list(B.FEATS),
            "round": args.round, "args": vars(args), "best_epoch": best[2] + 1}, path)
log(f"saved {path}")
