import sys, time, warnings, copy, numpy as np, pandas as pd, lightgbm as lgb, torch, torch.nn as nn
warnings.filterwarnings("ignore"); torch.set_num_threads(20); torch.manual_seed(0); np.random.seed(0)
sys.path.insert(0, '.'); import baseline as B
t0=time.time(); log=lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
KST="Asia/Seoul"; FROM=pd.Timestamp("2023-06-01 01:00",tz=KST); INNER=pd.Timestamp("2024-04-01 01:00",tz=KST); MID=pd.Timestamp("2024-06-01 01:00",tz=KST)
sites=pd.read_csv("sites.csv")[["site_id","current_group","capacity_kw","lat","lon","wa_station","wb_station"]]
gen=pd.read_parquet("train/generation.parquet"); gen["ts"]=pd.to_datetime(gen.ts); gen=gen[gen.ts>=FROM].reset_index(drop=True)
fa=pd.read_parquet("train/forecast_wa.parquet", columns=["wa_station","round","ts"]+B.WA_COLS)
fb=pd.read_parquet("train/forecast_wb.parquet", columns=["wb_station","round","ts"]+B.WB_COLS)
for d_ in (fa,fb): d_["ts"]=pd.to_datetime(d_.ts)
fa=fa[(fa.ts>=FROM)&(fa["round"]==17)]; fb=fb[(fb.ts>=FROM)&(fb["round"]==17)]
d=B.features(gen,sites,fa,fb,None,17); d=d[d.elev>0].reset_index(drop=True)
y=(d.energy/d.cap).clip(0,1.2).astype("float32").values; cap=d.cap.values.astype("float32"); ts=d.ts
tr_all=(ts<MID).values; fit=(ts<INNER).values; inner=((ts>=INNER)&(ts<MID)).values; va=(ts>=MID).values
log(f"rows {len(y):,} | GBDT/NN 학습 {tr_all.sum():,} (NN은 그중 {fit.sum():,} 학습, {inner.sum():,} 조기종료용) | 검증 {va.sum():,}")

yv=y[va]; cv=cap[va]; tsv=ts[va].values
def mae_kwh(p): return float((np.abs(p-yv)*cv).mean())
def agg_rel(p):
    g=pd.DataFrame({"ts":tsv,"p":p*cv,"a":yv*cv}).groupby("ts").sum(); return float(np.abs(g.p-g.a).sum()/g.a.sum())

# 기준: GBDT (L1, round 17) — 앞 실험과 동일
X=d[B.FEATS].astype("float32")
g=lgb.LGBMRegressor(n_estimators=300, learning_rate=0.08, num_leaves=63, min_child_samples=200, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.8, verbose=-1, n_jobs=-1, objective="l1").fit(X[tr_all], y[tr_all])
pg=np.clip(g.predict(X[va]),0,None); log(f"GBDT 기준            MAE {mae_kwh(pg):6.2f} kWh  합계 {agg_rel(pg)*100:5.2f}%")

# 피처 집합
def base_feats(df): return df[B.FEATS].astype("float32").values
def eng_feats(df):
    F=df[B.FEATS].astype("float32").copy()
    h=df.hour.values.astype("float32"); F["hour_sin"]=np.sin(2*np.pi*h/24); F["hour_cos"]=np.cos(2*np.pi*h/24)
    az=np.radians(df.azim.values.astype("float32")); F["az_sin"]=np.sin(az); F["az_cos"]=np.cos(az)
    F["log_rain"]=np.log1p(df.wa_rain.values); F["log_liq"]=np.log1p(df.wb_total_liq.values); F["log_cap"]=np.log(df.cap.values)
    F["miss_wa"]=df.wa_cloud.isna().astype("float32").values; F["miss_wb"]=df.wb_cloud.isna().astype("float32").values
    F["cs_wa"]=df.cosz.values*(1-df.wa_cloud.values/100); F["cs_wb"]=df.cosz.values*(1-df.wb_cloud.values/100); F["cs_uv"]=df.cosz.values*df.wb_uv_idx.values/12
    return F.astype("float32").values
def prep(F):
    med=np.nanmedian(F[fit],0); F=F.copy(); ii=np.where(np.isnan(F)); F[ii]=np.take(med,ii[1])
    mu=F[fit].mean(0); sd=F[fit].std(0)+1e-6; return np.clip((F-mu)/sd,-6,6).astype("float32")
# site 인덱스: 학습(fit)에 없는 발전소는 0('모르는 발전소') 토큰
present=set(d.site_id[fit].unique()); sid_map={s:(i+1 if s in present else 0) for i,s in enumerate(sites.site_id)}
sidx=d.site_id.map(sid_map).values.astype("int64")

class MLP(nn.Module):
    def __init__(s, nin, hidden, emb_dim=0, n_sites=254, drop=0.0):
        super().__init__(); s.emb=nn.Embedding(n_sites, emb_dim) if emb_dim else None
        layers=[]; k=nin+emb_dim
        for h in hidden: layers+=[nn.Linear(k,h), nn.ReLU()] + ([nn.Dropout(drop)] if drop else []); k=h
        layers+=[nn.Linear(k,1)]; s.net=nn.Sequential(*layers)
    def forward(s,x,si=None):
        if s.emb is not None: x=torch.cat([x,s.emb(si)],1)
        return s.net(x).squeeze(1)

def train_nn(name, F, loss="l1", hidden=(128,64), emb_dim=0, epochs=40, lr=2e-3, bs=4096, drop=0.0, subsample=None, early_stop=True, patience=6):
    Xa=prep(F); rng=np.random.default_rng(0)
    tr_idx=np.where(fit if early_stop else tr_all)[0]
    if subsample: tr_idx=rng.choice(tr_idx, subsample, replace=False)
    Xt=torch.tensor(Xa[tr_idx]); yt=torch.tensor(y[tr_idx]); st=torch.tensor(sidx[tr_idx])
    Xi=torch.tensor(Xa[inner]); yi=y[inner]; si=torch.tensor(sidx[inner]); ci=cap[inner]
    Xv=torch.tensor(Xa[va]); sv=torch.tensor(sidx[va])
    m=MLP(Xa.shape[1], hidden, emb_dim, drop=drop); opt=torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-5)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit=nn.L1Loss() if loss=="l1" else nn.MSELoss()
    best=(1e9,None,0); bad=0; n=len(tr_idx)
    for ep in range(epochs):
        m.train(); perm=torch.randperm(n)
        for i in range(0,n,bs):
            b=perm[i:i+bs]; sb=st[b].clone()
            if emb_dim: sb[torch.rand(len(sb))<0.05]=0          # 5%는 '모르는 발전소'로 바꿔 0번 임베딩도 학습
            opt.zero_grad(); l=crit(m(Xt[b], sb if emb_dim else None), yt[b]); l.backward(); opt.step()
        sched.step()
        if early_stop:
            m.eval()
            with torch.no_grad(): pi=np.clip(m(Xi, si if emb_dim else None).numpy(),0,None)
            mi=float((np.abs(pi-yi)*ci).mean())
            if mi<best[0]-1e-4: best=(mi,copy.deepcopy(m.state_dict()),ep); bad=0
            else:
                bad+=1
                if bad>=patience: break
    if early_stop and best[1] is not None: m.load_state_dict(best[1])
    m.eval()
    with torch.no_grad(): pv=np.clip(m(Xv, sv if emb_dim else None).numpy(),0,None)
    log(f"{name:22s} MAE {mae_kwh(pv):6.2f} kWh  합계 {agg_rel(pv)*100:5.2f}%  (epochs {ep+1}, best {best[2]+1 if early_stop else '-'})")
    return pv

Fb=base_feats(d); Fe=eng_feats(d)
P={}
P["V0 복제: MSE·40만행·40ep"]=train_nn("V0 복제", Fb, loss="mse", epochs=40, lr=1e-3, bs=2048, subsample=400_000, early_stop=False)
P["V1 MSE·전체·조기종료"]=train_nn("V1 MSE·전체·조기종료", Fb, loss="mse")
P["V2 +MAE 손실"]=train_nn("V2 +MAE 손실", Fb, loss="l1")
P["V3 +site 임베딩"]=train_nn("V3 +site 임베딩", Fb, loss="l1", emb_dim=8)
P["V4 +피처 가공"]=train_nn("V4 +피처 가공", Fe, loss="l1", emb_dim=8)
P["V5 +큰 네트워크"]=train_nn("V5 +큰 네트워크", Fe, loss="l1", emb_dim=16, hidden=(512,512,256), drop=0.1, epochs=60, lr=1e-3, patience=8)
best_name=min(P, key=lambda k: mae_kwh(P[k])); pb=P[best_name]
print(); print(f"가장 좋은 NN: {best_name}")
print(f"NN(best)와 GBDT 오차 상관: {np.corrcoef(pb-yv, pg-yv)[0,1]:.3f}")
for w in (0.5,0.3):
    pe=(1-w)*pg+w*pb; print(f"GBDT {1-w:.1f} + NN {w:.1f} 평균:  MAE {mae_kwh(pe):6.2f} kWh  합계 {agg_rel(pe)*100:5.2f}%")
print(f"GBDT 단독:               MAE {mae_kwh(pg):6.2f} kWh  합계 {agg_rel(pg)*100:5.2f}%")
