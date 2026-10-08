import sys, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '.')
import baseline as B   # 기준 코드의 solar/features/FEATS 재사용 (main은 실행되지 않음)

t0=time.time(); log=lambda m: print(f"[{time.time()-t0:5.1f}s] {m}", flush=True)
FROM=pd.Timestamp("2023-06-01 01:00", tz="Asia/Seoul"); MID=pd.Timestamp("2024-06-01 01:00", tz="Asia/Seoul")
sites=pd.read_csv("sites.csv")[["site_id","current_group","capacity_kw","lat","lon","wa_station","wb_station"]]
gen=pd.read_parquet("train/generation.parquet"); gen["ts"]=pd.to_datetime(gen.ts); gen=gen[gen.ts>=FROM]
fa=pd.read_parquet("train/forecast_wa.parquet", columns=["wa_station","round","ts"]+B.WA_COLS)
fb=pd.read_parquet("train/forecast_wb.parquet", columns=["wb_station","round","ts"]+B.WB_COLS)
for d in (fa,fb): d["ts"]=pd.to_datetime(d.ts)
fa=fa[(fa.ts>=FROM)&(fa["round"]==17)]; fb=fb[(fb.ts>=FROM)&(fb["round"]==17)]
d=B.features(gen,sites,fa,fb,None,17); d=d[d.elev>0].reset_index(drop=True)
d["y"]=(d.energy/d.cap).clip(0,1.2).astype("float32")
X=d[B.FEATS].astype("float32"); y=d.y.values; cap=d.cap.values
log(f"daylight rows {len(d):,}  ({d.ts.min():%Y-%m-%d} ~ {d.ts.max():%Y-%m-%d})")

rng=np.random.default_rng(0)
# A) 무작위 행 분할 50/50
mA=rng.random(len(d))<0.5
# C) 무작위 '날' 분할 50/50 (같은 날은 통째로 한쪽)
days=d.ts.dt.floor("D"); ud=days.unique(); pick=set(rng.choice(ud, size=len(ud)//2, replace=False)); mC=days.isin(pick).values
# B) 시간 분할: 앞 1년 학습, 뒤 1년 검증
mB=(d.ts<MID).values

def fit_eval(name, train_mask, objective="l2"):
    p=dict(n_estimators=300, learning_rate=0.08, num_leaves=63, min_child_samples=200, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, verbose=-1, n_jobs=-1, objective=objective)
    mdl=lgb.LGBMRegressor(**p).fit(X[train_mask], y[train_mask])
    v=~train_mask; pred=np.clip(mdl.predict(X[v]),0,None)
    mae_cf=np.abs(pred-y[v]).mean()
    err_kwh=np.abs(pred-y[v])*cap[v]; mae_kwh=err_kwh.mean()
    # 전체 발전소 합계(그 시각에 검증에 있는 발전소들) 기준 상대오차
    g=pd.DataFrame({"ts":d.ts.values[v],"p":pred*cap[v],"a":y[v]*cap[v]}).groupby("ts").sum()
    agg_rel=np.abs(g.p-g.a).sum()/g.a.sum()
    log(f"{name:28s} train {train_mask.sum():>9,}  val {v.sum():>9,} | MAE(이용률) {mae_cf:.4f} | MAE(kWh/발전소·시간) {mae_kwh:6.2f} | 합계 상대오차 {agg_rel*100:5.2f}%")
    return mae_cf

fit_eval("A 무작위 행 분할 (L2)", mA)
fit_eval("C 무작위 날 분할 (L2)", mC)
fit_eval("B 시간 분할 앞1년→뒤1년 (L2)", mB)
fit_eval("B 시간 분할 (L1=MAE 학습)", mB, objective="l1")
