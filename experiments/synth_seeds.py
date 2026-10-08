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
yv=y[va]; cv=cap[va]; tsv=ts[va].values
def mae_kwh(p): return float((np.abs(p-yv)*cv).mean())
def agg_rel(p):
    g=pd.DataFrame({"ts":tsv,"p":p*cv,"a":yv*cv}).groupby("ts").sum(); return float(np.abs(g.p-g.a).sum()/g.a.sum())

# 실측 기상 (학습 구간) → 예보 오차 표준편차, 그리고 '실측을 입력으로' 행
OA=["wa_cloud","wa_temp","wa_humidity","wa_rain","wa_wind_speed","wa_ground_press"]
OB=["wb_cloud","wb_temp","wb_rel_hum","wb_uv_idx","wb_vis","wb_wind_speed","wb_dew_point"]
oa=pd.read_parquet("train/observed_wa.parquet", columns=["wa_station","ts"]+OA); ob=pd.read_parquet("train/observed_wb.parquet", columns=["wb_station","ts"]+OB)
for o in (oa,ob): o["ts"]=pd.to_datetime(o.ts)
oa=oa[oa.ts>=FROM]; ob=ob[ob.ts>=FROM]
oa.loc[oa.wa_wind_speed>50,"wa_wind_speed"]=np.nan; ob.loc[ob.wb_cloud>100,"wb_cloud"]=np.nan; ob.loc[ob.wb_uv_idx>15,"wb_uv_idx"]=np.nan; ob.loc[ob.wb_vis>50,"wb_vis"]=np.nan; ob.loc[ob.wb_wind_speed>200,"wb_wind_speed"]=np.nan
obs=d[["site_id","ts","wa_station","wb_station"]].merge(oa.rename(columns={c:c+"_o" for c in OA}),on=["wa_station","ts"],how="left").merge(ob.rename(columns={c:c+"_o" for c in OB}),on=["wb_station","ts"],how="left")
noise_sd=np.zeros(len(B.FEATS),dtype="float32")
for j,c in enumerate(B.FEATS):
    if c in OA+OB:
        r=(obs[c+"_o"].values-d[c].values)[tr_all]; r=r[~np.isnan(r)]; noise_sd[j]=np.nanstd(r)
log("예보 오차 표준편차(실측−예보): "+", ".join(f"{c}={s:.1f}" for c,s in zip(B.FEATS,noise_sd) if s>0))
Xraw=d[B.FEATS].astype("float32").values
Xobs=Xraw.copy()
for j,c in enumerate(B.FEATS):
    if c in OA+OB:
        o=obs[c+"_o"].values.astype("float32"); m=~np.isnan(o); Xobs[m,j]=o[m]

present=set(d.site_id[fit].unique()); sid_map={s:(i+1 if s in present else 0) for i,s in enumerate(sites.site_id)}
sidx=d.site_id.map(sid_map).values.astype("int64")
med=np.nanmedian(Xraw[fit],0); mu=None; sd=None
def prep(F):
    global mu,sd
    F=F.copy(); ii=np.where(np.isnan(F)); F[ii]=np.take(med,ii[1])
    if mu is None: mu=F[fit].mean(0); sd=F[fit].std(0)+1e-6
    return np.clip((F-mu)/sd,-6,6).astype("float32")
Xs=prep(Xraw); Xso=prep(Xobs); noise_sd_std=(noise_sd/sd).astype("float32")

class MLP(nn.Module):
    def __init__(s, nin, hidden=(128,64), emb_dim=8, n_sites=254):
        super().__init__(); s.emb=nn.Embedding(n_sites, emb_dim); layers=[]; k=nin+emb_dim
        for h in hidden: layers+=[nn.Linear(k,h), nn.ReLU()]; k=h
        layers+=[nn.Linear(k,1)]; s.net=nn.Sequential(*layers)
    def forward(s,x,si): return s.net(torch.cat([x,s.emb(si)],1)).squeeze(1)

def train_nn(name, X_tr, y_tr, s_tr, w_tr=None, noise_scale=0.0, epochs=40, lr=2e-3, bs=4096, patience=6):
    Xt=torch.tensor(X_tr); yt=torch.tensor(y_tr); st=torch.tensor(s_tr); wt=torch.tensor(w_tr if w_tr is not None else np.ones(len(y_tr),dtype="float32"))
    Xi=torch.tensor(Xs[inner]); yi=y[inner]; si=torch.tensor(sidx[inner]); ci=cap[inner]; Xv=torch.tensor(Xs[va]); sv=torch.tensor(sidx[va])
    nz=torch.tensor(noise_sd_std)*noise_scale
    m=MLP(Xt.shape[1]); opt=torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-5); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    best=(1e9,None,0); bad=0; n=len(yt)
    for ep in range(epochs):
        m.train(); perm=torch.randperm(n)
        for i in range(0,n,bs):
            b=perm[i:i+bs]; sb=st[b].clone(); sb[torch.rand(len(sb))<0.05]=0; xb=Xt[b]
            if noise_scale>0: xb=xb+torch.randn_like(xb)*nz
            opt.zero_grad(); l=(wt[b]*torch.abs(m(xb,sb)-yt[b])).mean(); l.backward(); opt.step()
        sched.step(); m.eval()
        with torch.no_grad(): pi=np.clip(m(Xi,si).numpy(),0,None)
        mi=float((np.abs(pi-yi)*ci).mean())
        if mi<best[0]-1e-4: best=(mi,copy.deepcopy(m.state_dict()),ep); bad=0
        else:
            bad+=1
            if bad>=patience: break
    m.load_state_dict(best[1]); m.eval()
    with torch.no_grad(): pv=np.clip(m(Xv,sv).numpy(),0,None)
    log(f"{name:40s} MAE {mae_kwh(pv):6.2f} kWh  합계 {agg_rel(pv)*100:5.2f}%  (학습행 {n:,}, best ep {best[2]+1})")
    return pv


f=np.where(fit)[0]
res={"기준 (증강 없음)":[], "예보오차 노이즈 1.0σ":[], "실측 기상 입력 행 추가":[]}
for seed in (0,1,2):
    torch.manual_seed(seed); np.random.seed(seed)
    res["기준 (증강 없음)"].append(train_nn(f"seed{seed} 기준", Xs[f], y[f], sidx[f]))
    torch.manual_seed(seed); res["예보오차 노이즈 1.0σ"].append(train_nn(f"seed{seed} 노이즈1.0", Xs[f], y[f], sidx[f], noise_scale=1.0))
    torch.manual_seed(seed); res["실측 기상 입력 행 추가"].append(train_nn(f"seed{seed} 실측행", np.concatenate([Xs[f],Xso[f]]), np.concatenate([y[f],y[f]]), np.concatenate([sidx[f],sidx[f]])))
print()
print(f"{'NN 변형 (시드 3개)':28s} {'MAE 평균':>8s} {'MAE 범위':>14s} {'합계오차 평균':>9s} {'합계오차 범위':>14s}")
for k,ps in res.items():
    m=[mae_kwh(p) for p in ps]; a=[agg_rel(p)*100 for p in ps]
    print(f"{k:28s} {np.mean(m):8.2f} {min(m):6.2f}~{max(m):5.2f}  {np.mean(a):8.2f}%  {min(a):5.2f}~{max(a):5.2f}%")
sa=np.mean([p for p in res["예보오차 노이즈 1.0σ"]],0)
print()
print(f"노이즈 모델 3개 시드 평균 앙상블: MAE {mae_kwh(sa):6.2f}  합계 {agg_rel(sa)*100:5.2f}%")
