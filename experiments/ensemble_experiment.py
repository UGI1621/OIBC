import sys, time, warnings, numpy as np, pandas as pd, lightgbm as lgb
warnings.filterwarnings("ignore")
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
sys.path.insert(0, '.'); import baseline as B
t0=time.time(); log=lambda m: print(f"[{time.time()-t0:5.1f}s] {m}", flush=True)
FROM=pd.Timestamp("2023-06-01 01:00", tz="Asia/Seoul"); MID=pd.Timestamp("2024-06-01 01:00", tz="Asia/Seoul"); H2=pd.Timestamp("2024-12-01 01:00", tz="Asia/Seoul")
sites=pd.read_csv("sites.csv")[["site_id","current_group","capacity_kw","lat","lon","wa_station","wb_station"]]
gen=pd.read_parquet("train/generation.parquet"); gen["ts"]=pd.to_datetime(gen.ts); gen=gen[gen.ts>=FROM].reset_index(drop=True)
fa=pd.read_parquet("train/forecast_wa.parquet", columns=["wa_station","round","ts"]+B.WA_COLS)
fb=pd.read_parquet("train/forecast_wb.parquet", columns=["wb_station","round","ts"]+B.WB_COLS)
for d_ in (fa,fb): d_["ts"]=pd.to_datetime(d_.ts)
fa=fa[fa.ts>=FROM]; fb=fb[fb.ts>=FROM]
d17=B.features(gen,sites,fa,fb,None,17); d10=B.features(gen,sites,fa,fb,None,10)
assert (d17.site_id.values==d10.site_id.values).all() and (d17.ts.values==d10.ts.values).all()
day=(d17.elev>0).values; d17=d17[day].reset_index(drop=True); d10=d10[day].reset_index(drop=True)
y=(d17.energy/d17.cap).clip(0,1.2).astype("float32").values; cap=d17.cap.values; ts=d17.ts
X17=d17[B.FEATS].astype("float32"); X10=d10[B.FEATS].astype("float32")
tr=(ts<MID).values; va=~tr
log(f"rows {len(y):,}  train {tr.sum():,}  val {va.sum():,}")

def lgbm(X, objective="l1", seed=0, leaves=63, lr=0.08, n=300):
    m=lgb.LGBMRegressor(n_estimators=n, learning_rate=lr, num_leaves=leaves, min_child_samples=200, subsample=0.8, subsample_freq=1,
                        colsample_bytree=0.8, verbose=-1, n_jobs=-1, objective=objective, random_state=seed).fit(X[tr], y[tr])
    return np.clip(m.predict(X[va]),0,None)
P={}
P["G17 (GBDT, L1, round17)"]=lgbm(X17); log("G17")
P["G10 (GBDT, L1, round10)"]=lgbm(X10); log("G10")
P["G17b (seed·설정만 다름)"]=lgbm(X17, seed=7, leaves=31, lr=0.05, n=600); log("G17b")
P["G17-L2 (손실만 다름)"]=lgbm(X17, objective="l2"); log("G17-L2")
# 신경망 (다른 알고리즘 계열). 학습 표본 40만 행 부분추출로 시간 절약
rng=np.random.default_rng(0); idx=np.where(tr)[0]; sub=rng.choice(idx, 400_000, replace=False)
med=np.nanmedian(X17.values[tr], axis=0)          # 결측은 학습 구간 중앙값으로 채움 (신경망은 NaN 불가)
def fill(A): A=A.copy(); ii=np.where(np.isnan(A)); A[ii]=np.take(med, ii[1]); return A
sc=StandardScaler().fit(fill(X17.values[sub]))
mlp=MLPRegressor(hidden_layer_sizes=(128,64), max_iter=40, early_stopping=True, random_state=0, batch_size=2048, learning_rate_init=1e-3)
mlp.fit(sc.transform(fill(X17.values[sub])), y[sub]); P["NN (MLP, round17)"]=np.clip(mlp.predict(sc.transform(fill(X17.values[va]))),0,None); log("NN")

yv=y[va]; cv=cap[va]; tsv=ts[va].values
def mae_kwh(p): return (np.abs(p-yv)*cv).mean()
def agg_rel(p):
    g=pd.DataFrame({"ts":tsv,"p":p*cv,"a":yv*cv}).groupby("ts").sum(); return np.abs(g.p-g.a).sum()/g.a.sum()
names=list(P); E=np.column_stack([P[n]-yv for n in names])
print("\n=== 개별 모델 ===")
for n in names: print(f"{n:28s} MAE {mae_kwh(P[n]):6.2f} kWh  합계 상대오차 {agg_rel(P[n])*100:5.2f}%")
print("\n=== 모델 간 오차 상관 (낮을수록 섞을 때 이득) ===")
C=np.corrcoef(E.T); short=["G17","G10","G17b","G17-L2","NN"]
print(pd.DataFrame(C, index=short, columns=short).round(3).to_string())
print("\n=== 앙상블 ===")
def show(n,p): print(f"{n:44s} MAE {mae_kwh(p):6.2f} kWh  합계 상대오차 {agg_rel(p)*100:5.2f}%")
g17=P[names[0]]
show("G17 단독 (기준)", g17)
show("G17+G10 평균 (라운드 앙상블, 기준 코드 방식)", (g17+P[names[1]])/2)
show("G17+G17b 평균 (시드·설정 앙상블)", (g17+P[names[2]])/2)
show("G17+G17-L2 평균 (손실 앙상블)", (g17+P[names[3]])/2)
show("G17+NN 평균 (알고리즘 앙상블)", (g17+P[names[4]])/2)
allp=np.column_stack([P[n] for n in names])
show("5개 단순 평균", allp.mean(1))
show("5개 중앙값", np.median(allp,1))
# 가중 평균: 앞 반년으로 가중치 학습, 뒤 반년으로 평가 (OOS)
h1=(ts[va]<H2).values
h2=~h1
def fit_w(mask):
    def obj(w): w=np.abs(w)/np.abs(w).sum(); return (np.abs(allp[mask]@w-yv[mask])*cv[mask]).mean()
    r=minimize(obj, np.ones(5)/5, method="Nelder-Mead", options={"maxiter":2000,"xatol":1e-4,"fatol":1e-6}); w=np.abs(r.x)/np.abs(r.x).sum(); return w
w1=fit_w(h1); w2=fit_w(h2)
p_oos=np.where(h2, allp@w1, allp@w2)
show("5개 가중 평균 (반년 학습→반대 반년 적용, OOS)", p_oos)
print("가중치(앞 반년 학습):", dict(zip(short, w1.round(3))))
print("가중치(뒤 반년 학습):", dict(zip(short, w2.round(3))))
