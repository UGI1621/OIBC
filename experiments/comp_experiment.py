import sys, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '.'); import baseline as B
t0=time.time(); log=lambda m: print(f"[{time.time()-t0:5.1f}s] {m}", flush=True)
FROM=pd.Timestamp("2023-06-01 01:00", tz="Asia/Seoul"); MID=pd.Timestamp("2024-06-01 01:00", tz="Asia/Seoul"); H2=pd.Timestamp("2024-12-01 01:00", tz="Asia/Seoul")
sites=pd.read_csv("sites.csv")[["site_id","current_group","capacity_kw","lat","lon","wa_station","wb_station"]]
gen=pd.read_parquet("train/generation.parquet"); gen["ts"]=pd.to_datetime(gen.ts); gen=gen[gen.ts>=FROM]
fa=pd.read_parquet("train/forecast_wa.parquet", columns=["wa_station","round","ts"]+B.WA_COLS)
fb=pd.read_parquet("train/forecast_wb.parquet", columns=["wb_station","round","ts"]+B.WB_COLS)
for d_ in (fa,fb): d_["ts"]=pd.to_datetime(d_.ts)
fa=fa[(fa.ts>=FROM)&(fa["round"]==17)]; fb=fb[(fb.ts>=FROM)&(fb["round"]==17)]
d=B.features(gen,sites,fa,fb,None,17); d=d[d.elev>0].reset_index(drop=True)
d["y"]=(d.energy/d.cap).clip(0,1.2).astype("float32"); X=d[B.FEATS].astype("float32")
tr=(d.ts<MID).values
mdl=lgb.LGBMRegressor(n_estimators=300, learning_rate=0.08, num_leaves=63, min_child_samples=200, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.8, verbose=-1, n_jobs=-1, objective="l1").fit(X[tr], d.y.values[tr])
v=d[~tr].copy(); v["pred_kwh"]=np.clip(mdl.predict(X[~tr]),0,None)*v.cap.values; v["err"]=v.pred_kwh-v.energy
log(f"val rows {len(v):,}  site-hour MAE {np.abs(v.err).mean():.2f} kWh")

# 오차 행렬 E (시각 × 발전소), 가중치 w = SMP×k
E=v.pivot_table(index="ts", columns="site_id", values="err", aggfunc="sum").reindex(columns=sites.site_id).fillna(0.0)
A=v.pivot_table(index="ts", columns="site_id", values="energy", aggfunc="sum").reindex(columns=sites.site_id).fillna(0.0)
smp=pd.read_csv("train/smp.csv"); smp["ts"]=pd.to_datetime(smp.ts); smp=smp.set_index("ts").reindex(E.index)
hr=E.index.tz_convert("Asia/Seoul").hour; k=np.where((hr>=11)&(hr<=15),3,1); w=(smp.smp.values*k)
Em=E.values; cap=sites.capacity_kw.values; T=len(Em)
revenue=(smp.smp.values*A.values.sum(1)).sum()
log(f"val hours {T:,}  revenue(SMP×actual) {revenue/1e8:.2f}억원  k=3 share of w {w[k==3].sum()/w.sum():.3f}")

def penalty(m1, idx=None):
    s=slice(None) if idx is None else idx
    s1=Em[s]@m1; s2=Em[s].sum(1)-s1
    return 0.1*(w[s]*(np.abs(s1)+np.abs(s2))).sum()
one_group=0.1*(w*np.abs(Em.sum(1))).sum()
def feasible(m1): c1=cap[m1==1].sum(); return c1>=20000 and cap.sum()-c1>=20000

def by_sort(key):
    order=np.argsort(key); m=np.zeros(253); c=0
    for i in order:
        if c>=cap.sum()/2: break
        m[i]=1; c+=cap[i]
    assert feasible(m); return m
ref=(sites.current_group=="A").astype(float).values
ew=by_sort(sites.lon.values); ns=by_sort(sites.lat.values)
rng=np.random.default_rng(1)
def rand_part():
    while True:
        m=by_sort(rng.random(253))
        if feasible(m): return m
rands=[rand_part() for _ in range(30)]

def local_search(m, idx, iters=6000):
    m=m.copy(); best=penalty(m,idx)
    for _ in range(iters):
        if rng.random()<0.5:
            i=rng.integers(253); m2=m.copy(); m2[i]=1-m2[i]
        else:
            g1=np.where(m==1)[0]; g2=np.where(m==0)[0]; i=rng.choice(g1); j=rng.choice(g2); m2=m.copy(); m2[i]=0; m2[j]=1
        if not feasible(m2): continue
        p=penalty(m2,idx)
        if p<best: best, m = p, m2
    return m
h1=np.where(E.index<H2)[0]; h2=np.where(E.index>=H2)[0]
start=min(rands, key=penalty)
opt_on_h1=local_search(start,h1); opt_on_h2=local_search(start,h2); opt_full=local_search(start,None)
oos=penalty(opt_on_h1,h2)+penalty(opt_on_h2,h1)
log("local search done")

rows=[("한 팀으로 묶었다면 (불가능, 하한)", one_group),
      ("참조 배정 A/B", penalty(ref)),
      ("동서 분할 (경도 순)", penalty(ew)),
      ("남북 분할 (위도 순)", penalty(ns)),
      ("무작위 균형 분할 (30회 평균)", np.mean([penalty(m) for m in rands])),
      ("무작위 균형 분할 (30회 중 최악)", max(penalty(m) for m in rands)),
      ("최적화 분할 (반년 최적화→나머지 반년 평가, OOS)", oos),
      ("최적화 분할 (1년 전체로 최적화, 과적합 포함)", penalty(opt_full))]
print()
print(f"{'분할':48s} {'벌금(억원)':>10s} {'매출 대비':>9s} {'하한 대비':>9s}")
for n,p in rows: print(f"{n:48s} {p/1e8:10.3f} {p/revenue*100:8.2f}% {(p/one_group-1)*100:+8.1f}%")
g1c=cap[opt_full==1].sum(); print(f"\n최적화 분할 G1 용량 {g1c:,.0f} kW / G2 {cap.sum()-g1c:,.0f} kW, G1 개소 {int(opt_full.sum())}")
# 두 팀 오차의 상관 (왜 차이가 나는지)
for n,m in [("참조",ref),("동서",ew),("남북",ns),("최적(전체)",opt_full)]:
    s1=Em@m; s2=Em.sum(1)-s1; print(f"{n:8s} 두 팀 시간별 오차 상관 {np.corrcoef(s1,s2)[0,1]:.3f}  부호 같은 시각 비율 {(np.sign(s1)==np.sign(s2)).mean():.3f}")
