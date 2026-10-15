"""MLP 프로젝트 공통 모듈: 경로, 데이터 로드, 피처, 전처리(Scaler), 모델 정의.

데이터 폴더(OIBC_2026_DATA/...)는 읽기만 한다. 모든 출력은 mlp/outputs/ 에 쓴다.
피처는 주최 측 baseline.py 의 features()/FEATS 를 그대로 가져와 기존 실험(experiments/)과
같은 입력을 쓴다. 입력층 구성은 docs 의 artifact "OIBC MLP 입력층" 참고.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "OIBC_2026_DATA" / "handover 2" / "01_participant_dataset"
OUT = ROOT / "mlp" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
KST = "Asia/Seoul"

# 주최 측 baseline.py 의 피처 함수를 가져온다. 데이터 폴더에 pyc 를 남기지 않는다.
sys.dont_write_bytecode = True
sys.path.insert(0, str(DATA))
import baseline as B  # noqa: E402

_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:6.1f}s] {msg}", flush=True)


def load_sites() -> pd.DataFrame:
    cols = ["site_id", "current_group", "capacity_kw", "lat", "lon", "wa_station", "wb_station"]
    return pd.read_csv(DATA / "sites.csv")[cols]


def load_forecasts(split: str, rnd: int, ts_from=None):
    fa = pd.read_parquet(DATA / split / "forecast_wa.parquet",
                         columns=["wa_station", "round", "ts"] + B.WA_COLS)
    fb = pd.read_parquet(DATA / split / "forecast_wb.parquet",
                         columns=["wb_station", "round", "ts"] + B.WB_COLS)
    for d in (fa, fb):
        d["ts"] = pd.to_datetime(d.ts)
    fa = fa[fa["round"] == rnd]
    fb = fb[fb["round"] == rnd]
    if ts_from is not None:
        fa = fa[fa.ts >= ts_from]
        fb = fb[fb.ts >= ts_from]
    return fa, fb


def build_train_frame(ts_from, rnd: int):
    """학습용 (발전소, 시각) 행렬. 주간(elev>0)만 남기고 타깃 y = 이용률."""
    sites = load_sites()
    gen = pd.read_parquet(DATA / "train" / "generation.parquet")
    gen["ts"] = pd.to_datetime(gen.ts)
    gen = gen[gen.ts >= ts_from].reset_index(drop=True)
    fa, fb = load_forecasts("train", rnd, ts_from)
    d = B.features(gen, sites, fa, fb, None, rnd)
    d = d[d.elev > 0].reset_index(drop=True)
    d["y"] = (d.energy / d.cap).clip(0, 1.2).astype("float32")
    return d, sites


def build_test_frame(rnd: int):
    """평가 구간 (발전소 253 × 시각 8,760) 행렬. smp.csv 의 ts 가 채점 시간축."""
    sites = load_sites()
    index = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(DATA / "smp.csv").ts).dt.tz_convert(KST))
    fa, fb = load_forecasts("test", rnd)
    d = B.features(None, sites, fa, fb, index, rnd)
    return d, sites, index


class Scaler:
    """학습 행에서 중앙값·평균·표준편차를 구해 저장하고, 어느 구간에나 같은 값으로 적용한다.
    순서: NaN -> 중앙값, (x-평균)/표준편차, ±6 clip."""

    def fit(self, F: np.ndarray) -> "Scaler":
        self.med = np.nanmedian(F, 0)
        G = self._fill(F)
        self.mu = G.mean(0)
        self.sd = G.std(0) + 1e-6
        return self

    def _fill(self, F: np.ndarray) -> np.ndarray:
        F = F.copy()
        ii = np.where(np.isnan(F))
        F[ii] = np.take(self.med, ii[1])
        return F

    def transform(self, F: np.ndarray) -> np.ndarray:
        return np.clip((self._fill(F) - self.mu) / self.sd, -6, 6).astype("float32")

    def state(self) -> dict:
        return {"med": self.med, "mu": self.mu, "sd": self.sd}

    @classmethod
    def from_state(cls, s: dict) -> "Scaler":
        o = cls()
        o.med, o.mu, o.sd = s["med"], s["mu"], s["sd"]
        return o


class MLP(nn.Module):
    """수치 입력 + 발전소 임베딩 -> 은닉층(ReLU) -> 출력 1개(이용률). bias 는 Linear 기본값."""

    def __init__(self, nin: int, hidden=(128, 64), emb_dim: int = 8, n_sites: int = 254, drop: float = 0.0):
        super().__init__()
        self.emb = nn.Embedding(n_sites, emb_dim) if emb_dim else None
        layers = []
        k = nin + emb_dim
        for h in hidden:
            layers += [nn.Linear(k, h), nn.ReLU()] + ([nn.Dropout(drop)] if drop else [])
            k = h
        layers += [nn.Linear(k, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, si=None):
        if self.emb is not None:
            x = torch.cat([x, self.emb(si)], 1)
        return self.net(x).squeeze(1)


def site_index(sites: pd.DataFrame, known: set) -> dict:
    """sites.csv 순서대로 1..253. 학습에 없던 발전소는 0('모르는 발전소' 토큰)."""
    return {s: (i + 1 if s in known else 0) for i, s in enumerate(sites.site_id)}
