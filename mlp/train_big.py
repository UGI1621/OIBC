"""큰 신경망 학습 (공부용 단일 파일). 메인 모델.

  python mlp/train_big.py              # 홀드아웃 1년(2024-06~2025-05)으로 테스트 점수 확인
  python mlp/train_big.py --final      # 학습 구간 전체로 학습해 제출용 모델 저장

train_small.py 와 뼈대가 같고, 맨 위 CONFIG 만 다르다.
데이터 이름: 학습(train) / 검증(val, 조기 종료용 두 달) / 테스트(test, 홀드아웃 1년).
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import B, KST, OUT, Scaler, build_train_frame, log, mlp_predict, site_index

# ----------------------------------------------------------------------------- 설정
# 하이퍼파라미터. 값을 바꾸면 파일 이름(name)도 같이 바꿔 두면 산출물이 섞이지 않는다.
CONFIG = dict(
    name="big",

    # 은닉층 구조. 튜플 원소 하나가 층 하나, 숫자가 그 층의 뉴런 수.
    #   입력 24(수치) + 16(임베딩) = 40 -> 512 -> 512 -> 256 -> 1(출력).
    #   파라미터 42만 개(작은 망의 29배), 에폭당 11초(최종 모드 34초).
    #   더 넓히거나 더하면 과적합이 더 빨라지고 시간만 늘 가능성이 크다. 4층 이상은 잔차 연결 없이는 보통 득이 없다.
    #   관찰: 테스트 MAE 12.35 (작은 망 12.59). 차이 0.24는 시드 편차(0.23)와 같은 크기라 아직 확정 아님.
    hidden=(512, 512, 256),

    # 발전소 임베딩 차원. 발전소 253개 각각에 이 길이의 벡터를 배워 입력에 붙인다.
    #   발전소별 편향·성향을 흡수하는 역할. 큰 망에 맞춰 16. 기존 실험에서 8 -> 16은 단독 효과가 분리되지 않았다.
    emb_dim=16,

    # 드롭아웃 비율. 학습 중 각 은닉 뉴런을 이 확률로 무작위로 꺼서 과적합을 늦춘다.
    #   큰 망은 외울 용량이 커서 0.1을 건다. 0.2~0.3까지 올려볼 수 있으나 검증 MAE 흔들림이 커진다.
    #   드롭아웃은 model.train() 에서만 켜지고 model.eval() 에서는 꺼진다 (val_mae, 예측 때).
    dropout=0.1,

    # 시작 학습률. 가중치를 한 번에 얼마나 옮길지.
    #   코사인 스케줄이 이 값에서 시작해 epochs 에 걸쳐 0까지 줄인다. 조기 종료로 일찍 멈추면 아직 큰 상태.
    #   큰 망은 작은 망(2e-3)의 절반. 관찰: 그래도 검증 MAE가 11.3 -> 11.5 -> 11.0 -> 11.0 -> 11.3 으로 흔들렸다.
    #   더 낮추면(5e-4) 흔들림은 줄지만 에폭을 더 돌아야 한다.
    lr=1e-3,

    # 가중치 감쇠(L2 규제). 매 갱신마다 가중치를 이 비율만큼 0 쪽으로 당긴다.
    #   1e-5 는 사실상 거의 안 거는 수준. 큰 망의 과적합을 더 누르려면 1e-4 ~ 1e-3 도 시도해 볼 만하다.
    weight_decay=1e-5,

    # 미니배치 크기. 한 번의 가중치 갱신에 쓰는 행 수.
    #   학습 476,111행 / 4096 = 에폭당 117번 갱신. 작게 하면 갱신이 잦고 흔들림이 커지며 느려진다.
    #   크게 하면 빠르지만 같은 에폭으로 덜 배우고 lr 을 같이 키워야 한다. 4배 키우면 lr 2~4배.
    batch_size=4096,

    # 최대 에폭 수. 조기 종료가 먼저 걸리면 거기서 끝난다.
    #   코사인 스케줄의 길이이기도 해서, 이 값을 바꾸면 학습률이 줄어드는 속도도 바뀐다.
    #   관찰: 테스트 모드 11에폭(최적 3), 최종 모드 10에폭(최적 2)에서 멈춰 60 근처까지 간 적은 없다.
    epochs=60,

    # 조기 종료 인내. 검증 MAE 최저가 이만큼 연속 에폭 동안 갱신되지 않으면 멈추고 최저 시점 가중치로 되돌린다.
    #   드롭아웃 때문에 검증 MAE가 더 흔들려 작은 망(6)보다 길게. 검증 세트가 두 달치라 10~12도 고려.
    #   올리면 에폭당 11~34초라 시간이 그만큼 는다.
    patience=8,

    # 에폭 평균. N 이면 조기 종료 없이 N 에폭 학습하고, 매 에폭 가중치를 모두 저장해 예측 때 N 개 예측을 평균한다.
    #   0 이면 예전 방식(검증 MAE 최저 에폭 하나 + 조기 종료, patience 사용).
    #   이유(2026-10-28): 홀드아웃 벌금이 한 에폭마다 ±2,000만 원 출렁이고, 검증 MAE(두 달)와 벌금의 상관이 0.16~0.23뿐이라
    #   검증으로 에폭 하나를 고르는 게 운에 가깝다. 16 에폭 평균으로 MLP 3시드 벌금 204.5 -> 200.1백만, 혼합 201.3 -> 197.4백만.
    #   이 모드에서도 검증 MAE 는 기록만 한다. 학습 구간은 예전과 같다(검증 두 달은 학습에 안 씀).
    #   비용: 학습 시간이 16 에폭 고정, 모델 파일이 16배(약 27MB), 예측도 16번.
    epoch_avg=16,

    # 사용할 예보 발행 라운드. 10 = 전일 10시, 17 = 전일 17시. 둘 다 예측 시점(전일 17시) 이전이라 허용.
    #   17이 더 최신이라 기본. 베이스라인은 두 라운드 모델을 평균한다.
    round=17,

    # 학습 시작일. 발전량 데이터가 있는 2021-01부터 전부 쓴다 (학습 행 47.6만 -> 95.5만).
    #   베이스라인의 2023-06(최근 2년)보다 홀드아웃 벌금 -3.5% (3시드 모두 -3.4~-3.6%), 편향 -2.1% -> -0.5%.
    #   2021년은 31개소뿐이지만 날씨-발전량 관계를 더 배워 새 발전소 MAE 도 줄었다. 48개소는 2024-06 이후에만 데이터가 있다.
    train_from="2021-01-01",
)
# 코드 안에 박힌 값들(바꿀 일은 적지만 알아둘 것):
#   - 5%: 배치마다 발전소 번호의 5%를 0번('모르는 발전소')으로 바꿔 0번 임베딩도 학습시킨다 (학습 루프).
#   - 1e-4: 검증 MAE가 이보다 작게 좋아진 건 '갱신'으로 치지 않는다 (조기 종료 판단, epoch_avg=0 일 때만).
#   - ±6: 전처리에서 표준화된 값을 자르는 범위 (common.Scaler).
#   - 0~1.2: 타깃 이용률의 clip 범위 (common.build_train_frame).
#   - 2024-04-01 / 2024-06-01: 검증·테스트 분할 경계. --final 이면 2025-04-01 부터 검증.
# ------------------------------------------------------------------------------------

ap = argparse.ArgumentParser()
ap.add_argument("--final", action="store_true", help="학습 구간 전체로 학습 (제출용)")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
C = CONFIG

# 1. 데이터 ---------------------------------------------------------------------------
# 행 하나 = (발전소, 시각). 주간(elev>0)만. 타깃 y = 발전량 / 설비용량 (0~1.2)
d, sites = build_train_frame(pd.Timestamp(f"{C['train_from']} 01:00", tz=KST), C["round"])
ts = d.ts
if args.final:
    VAL_FROM, TEST_FROM = pd.Timestamp("2025-04-01 01:00", tz=KST), None
else:
    VAL_FROM, TEST_FROM = pd.Timestamp("2024-04-01 01:00", tz=KST), pd.Timestamp("2024-06-01 01:00", tz=KST)
train = (ts < VAL_FROM).values
val = ((ts >= VAL_FROM) & (ts < TEST_FROM)).values if TEST_FROM is not None else (ts >= VAL_FROM).values
test = (ts >= TEST_FROM).values if TEST_FROM is not None else np.zeros(len(d), bool)
log(f"[{C['name']}] hidden={C['hidden']} emb={C['emb_dim']} drop={C['dropout']} lr={C['lr']} "
    f"epoch_avg={C['epoch_avg']} seed={args.seed}")
log(f"rows {len(d):,} | train {train.sum():,} | val {val.sum():,} | test {test.sum():,}")

# 2. 전처리 ---------------------------------------------------------------------------
# 수치 24개: 학습 행 통계로 NaN->중앙값, 표준화, ±6 clip. 임베딩 인덱스는 그대로.
F = d[B.FEATS].astype("float32").values
y = d.y.values.astype("float32")
cap = d.cap.values.astype("float32")
scaler = Scaler().fit(F[train])
X = scaler.transform(F)
smap = site_index(sites, set(d.site_id[train].unique()))   # 학습에 없던 발전소는 0번 토큰
sidx = d.site_id.map(smap).values.astype("int64")


def tensors(mask):
    return torch.tensor(X[mask]), torch.tensor(y[mask]), torch.tensor(sidx[mask])


Xt, yt, st = tensors(train)
Xv, yv, sv = tensors(val)
cv = cap[val]


# 3. 모델 -----------------------------------------------------------------------------
class Net(nn.Module):
    """수치 입력(24) + 발전소 임베딩(emb_dim) -> [Linear -> ReLU -> Dropout] x 층 수 -> Linear -> 1.
    속성 이름(emb, net)은 common.MLP 와 같게 두어 predict.py 가 그대로 불러온다."""

    def __init__(self, n_in, hidden, emb_dim, dropout, n_sites=254):
        super().__init__()
        self.emb = nn.Embedding(n_sites, emb_dim)          # 발전소별로 배우는 보정 벡터
        layers, k = [], n_in + emb_dim
        for h in hidden:
            layers += [nn.Linear(k, h), nn.ReLU()]           # bias 는 Linear 기본값(True)
            if dropout:
                layers.append(nn.Dropout(dropout))           # 학습 때만 뉴런 10%를 무작위로 끈다
            k = h
        layers.append(nn.Linear(k, 1))                       # 출력 1개 = 이용률, 활성함수 없음
        self.net = nn.Sequential(*layers)

    def forward(self, x, site):
        return self.net(torch.cat([x, self.emb(site)], 1)).squeeze(1)


model = Net(X.shape[1], C["hidden"], C["emb_dim"], C["dropout"])
n_params = sum(p.numel() for p in model.parameters())
log(f"parameters {n_params:,}")
opt = torch.optim.AdamW(model.parameters(), lr=C["lr"], weight_decay=C["weight_decay"])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C["epochs"])
loss_fn = nn.L1Loss()                                       # 벌금이 |예측-실적| 이므로 L1


def val_mae(m):
    m.eval()                                                # eval 모드: 드롭아웃 꺼짐
    with torch.no_grad():
        p = np.clip(m(Xv, sv).numpy(), 0, None)
    return float((np.abs(p - yv.numpy()) * cv).mean())      # kWh 단위


# 4. 학습 루프 ------------------------------------------------------------------------
# epoch_avg 면 정해진 에폭 수만큼 돌며 매 에폭 가중치를 모은다. 코사인 스케줄 길이는 그대로 epochs(60).
best = dict(mae=np.inf, state=None, epoch=0)
bad, n, history, states = 0, len(Xt), [], []
for ep in range(1, (C["epoch_avg"] or C["epochs"]) + 1):
    model.train()                                            # train 모드: 드롭아웃 켜짐
    perm = torch.randperm(n)                                 # 에폭마다 새로 섞는다
    tot = 0.0
    for i in range(0, n, C["batch_size"]):                   # 미니배치 한 바퀴
        b = perm[i:i + C["batch_size"]]
        site_b = st[b].clone()
        site_b[torch.rand(len(site_b)) < 0.05] = 0           # 5%는 '모르는 발전소'로 바꿔 0번 임베딩도 학습
        opt.zero_grad()
        loss = loss_fn(model(Xt[b], site_b), yt[b])          # 순전파 + 손실
        loss.backward()                                      # 역전파
        opt.step()                                           # 가중치 갱신
        tot += loss.item() * len(b)
    sched.step()                                             # 학습률 한 단계 감소
    mae = val_mae(model)
    history.append(dict(epoch=ep, train_l1=tot / n, val_mae=mae, lr=opt.param_groups[0]["lr"]))
    log(f"epoch {ep:2d}  train L1 {tot / n:.4f}  val MAE {mae:.3f} kWh")
    if C["epoch_avg"]:                                       # 에폭 평균: 고르지 않고 전부 모은다
        states.append(copy.deepcopy(model.state_dict()))
        continue
    if mae < best["mae"] - 1e-4:                             # 좋아졌으면 가중치 복사해 두기
        best = dict(mae=mae, state=copy.deepcopy(model.state_dict()), epoch=ep)
        bad = 0
    else:
        bad += 1
        if bad >= C["patience"]:
            log(f"early stop at epoch {ep}")
            break
if C["epoch_avg"]:
    log(f"epoch average of {len(states)} epochs")
else:
    states = [best["state"]]                                 # 최저 시점 가중치 하나
    log(f"best epoch {best['epoch']}  val MAE {best['mae']:.3f} kWh")

# 5. 테스트 점수 (홀드아웃 1년, --final 이 아닐 때만) ----------------------------------
if test.any():
    ys, cs = y[test], cap[test]
    p = mlp_predict({"nin": X.shape[1], "hidden": C["hidden"], "emb": C["emb_dim"], "drop": C["dropout"],
                     "states": states}, X[test], sidx[test])   # 에폭마다 예측해 평균 (하나면 그대로)
    mae = float((np.abs(p - ys) * cs).mean())
    g = pd.DataFrame({"ts": ts[test].values, "p": p * cs, "a": ys * cs}).groupby("ts").sum()
    rel = float(np.abs(g.p - g.a).sum() / g.a.sum())
    log(f"TEST 2024-06~2025-05  MAE {mae:.2f} kWh/site-hour  팀 합계 상대오차 {rel * 100:.2f}%")

# 6. 저장 -----------------------------------------------------------------------------
tag = "final" if args.final else "test"
path = OUT / f"model_{C['name']}_{tag}_s{args.seed}.pt"
# states = 예측에 쓸 가중치 목록 (에폭 평균이면 N 개, 아니면 1개). common.mlp_predict 가 읽는다.
torch.save({"states": states, "hidden": tuple(C["hidden"]), "emb": C["emb_dim"], "drop": C["dropout"],
            "nin": X.shape[1], "scaler": scaler.state(), "smap": smap, "feats": list(B.FEATS),
            "round": C["round"], "config": C, "seed": args.seed,
            "best_epoch": None if C["epoch_avg"] else best["epoch"]}, path)
(OUT / f"history_{C['name']}_{tag}_s{args.seed}.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
log(f"saved {path}")
