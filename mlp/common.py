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
                         columns=["wa_station", "round", "ts"] + B.WA_COLS + EXTRA_WA)
    fb = pd.read_parquet(DATA / split / "forecast_wb.parquet",
                         columns=["wb_station", "round", "ts"] + B.WB_COLS + EXTRA_WB)
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


# 라운드 변화 입력: 같은 시각 예보의 (라운드 17 - 라운드 10). 예보가 전날 10시 -> 17시 사이에 바뀐 정도.
#   GBDT 는 '바뀐 예보를 다 믿지 않는' 패턴을 배워 홀드아웃 벌금 -1.0% (3시드 모두 감소). MLP 는 효과 없음.
RC = ["wa_cloud", "wb_cloud", "wa_rain", "wb_precip_prob", "wb_total_liq"]
RC_FEATS = [f"r_{c}" for c in RC]


def add_round_change(d: pd.DataFrame, d10: pd.DataFrame) -> pd.DataFrame:
    """d(라운드 17 프레임)에 RC_FEATS 열을 붙인다. d10 은 같은 구간의 라운드 10 프레임. 키 (site_id, ts)."""
    m = d[["site_id", "ts"]].merge(d10[["site_id", "ts"] + RC], on=["site_id", "ts"], how="left")
    for c in RC:
        d[f"r_{c}"] = d[c].values - m[c].values
    return d


# 풍향·눈 입력 (GBDT 전용): 베이스라인이 안 쓰던 예보 열 중 효과가 확인된 것만. load_forecasts 가 항상 읽어 둔다.
#   풍향 sin/cos 4개: GBDT 홀드아웃 벌금 -1.9% (3시드 모두 감소). MLP 에 넣으면 +4.2% 라 MLP 에는 안 넣는다.
#   눈·결빙 5개: 1년 전체로는 0%지만 눈 예보일(15일)만 보면 -4.2%, 다른 날 손해 없음. 눈 많은 겨울 대비.
#   나머지(운고·돌풍·체감/습구온도·비 세부·지점 거리)와 새 열의 라운드 변화는 효과가 없어 뺐다.
EXTRA_WA = ["wa_wind_dir", "wa_snow"]
EXTRA_WB = ["wb_wind_dir", "wb_snow_prob", "wb_ice_prob", "wb_snow", "wb_ice"]
WS_FEATS = ["wa_wind_dir_sin", "wa_wind_dir_cos", "wb_wind_dir_sin", "wb_wind_dir_cos",
            "wa_snow", "wb_snow_prob", "wb_ice_prob", "wb_snow", "wb_ice"]


# 이웃 시각·하루 요약 입력 (실험 브랜치 ugi/exp-neighbor-hours). 모두 같은 라운드 예보라 예측 마감 전에 이미 있는 값이다.
#   이웃 시각: 같은 지점 예보의 1시간 전·후 값. 예보가 시간을 조금 어긋나게 맞히는 오차를 줄이려는 것.
#   하루 요약: 그날 8~17시 예보의 구름 평균·표준편차, 강수 시간 수, 최대 강수확률. 종일 흐림인지, 구름이 오락가락하는지.
NB_COLS = {"wa": ["wa_cloud", "wa_rain"], "wb": ["wb_cloud", "wb_precip_prob"]}
NB_FEATS = ([f"{c}_{k}" for cs in NB_COLS.values() for c in cs for k in ("m1", "p1")]
            + ["day_wa_cloud_mean", "day_wa_cloud_std", "day_wb_cloud_mean", "day_wb_cloud_std",
               "day_wa_rain_hours", "day_wb_precip_prob_max"])


def add_neighbors(d: pd.DataFrame, split: str, rnd: int) -> pd.DataFrame:
    """d 에 NB_FEATS 를 붙인다. 지점별 예보 전체(밤 포함)에서 계산해 (지점, ts) 와 (지점, 날짜) 로 붙인다."""
    fa, fb = load_forecasts(split, rnd)
    for pre, f in (("wa", fa), ("wb", fb)):
        st, cols = f"{pre}_station", NB_COLS[pre]
        f = f[[st, "ts"] + cols]
        for k, h in (("m1", -1), ("p1", 1)):                             # ts 의 h 시간 뒤 예보를 ts 에 붙인다
            s = f.assign(ts=f.ts - pd.Timedelta(hours=h)).rename(columns={c: f"{c}_{k}" for c in cols})
            d = d.merge(s, on=[st, "ts"], how="left")
        day = f[f.ts.dt.hour.between(8, 17)].assign(date=lambda x: x.ts.dt.date)
        g = day.groupby([st, "date"])
        cl = f"{pre}_cloud"
        agg = pd.DataFrame({f"day_{cl}_mean": g[cl].mean(), f"day_{cl}_std": g[cl].std()})
        if pre == "wa":
            agg["day_wa_rain_hours"] = g.wa_rain.apply(lambda r: float((r > 0).sum()))
        else:
            agg["day_wb_precip_prob_max"] = g.wb_precip_prob.max()
        d = d.assign(date=d.ts.dt.date).merge(agg.reset_index(), on=[st, "date"], how="left").drop(columns="date")
    return d


def add_wind_snow(d: pd.DataFrame) -> pd.DataFrame:
    """풍향(도)을 sin/cos 로 바꿔 WS_FEATS 를 갖춘다. 359도와 1도가 가깝게 보이도록."""
    for c in ("wa_wind_dir", "wb_wind_dir"):
        r = np.deg2rad(d[c])
        d[c + "_sin"], d[c + "_cos"] = np.sin(r), np.cos(r)
    return d


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


def mlp_predict(ck: dict, X: np.ndarray, sidx: np.ndarray, chunk: int = 262144) -> np.ndarray:
    """체크포인트 하나로 예측(이용률, 음수는 0). 'states'(에폭 평균 모델)가 있으면 에폭마다 예측해 평균한다."""
    model = MLP(ck["nin"], ck["hidden"], ck["emb"], drop=ck.get("drop", 0.0))
    model.eval()
    preds = []
    for state in ck.get("states") or [ck["state"]]:
        model.load_state_dict(state)
        with torch.no_grad():
            p = np.concatenate([model(torch.tensor(X[i:i + chunk]), torch.tensor(sidx[i:i + chunk])).numpy()
                                for i in range(0, len(X), chunk)])
        preds.append(np.clip(p, 0, None))
    return np.mean(preds, axis=0)


def site_index(sites: pd.DataFrame, known: set) -> dict:
    """sites.csv 순서대로 1..253. 학습에 없던 발전소는 0('모르는 발전소' 토큰)."""
    return {s: (i + 1 if s in known else 0) for i, s in enumerate(sites.site_id)}
