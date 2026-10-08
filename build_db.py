import duckdb, os, time
ROOT = r"c:\code\OIBC"
D = os.path.join(ROOT, "OIBC_2026_DATA", "handover 2", "01_participant_dataset").replace("\\", "/")
DB = os.path.join(ROOT, "oibc.duckdb")
if os.path.exists(DB): os.remove(DB)
con = duckdb.connect(DB)
con.execute("SET TimeZone='Asia/Seoul'")
t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.1f}s] {m}", flush=True)

# ts: 원본은 tz-aware(KST). KST naive TIMESTAMP로 저장 (보기/조인 편의). 우측닫힘.
KST = "CAST(ts AT TIME ZONE 'Asia/Seoul' AS TIMESTAMP) AS ts"

con.execute(f"CREATE TABLE sites AS SELECT * FROM read_csv('{D}/sites.csv')")
con.execute(f"""CREATE TABLE smp AS
  SELECT {KST}, smp, k, 'test' AS split FROM read_csv('{D}/smp.csv', types={{'ts':'TIMESTAMPTZ'}})
  UNION ALL
  SELECT {KST}, smp, CASE WHEN hour(ts AT TIME ZONE 'Asia/Seoul') BETWEEN 11 AND 15 THEN 3 ELSE 1 END AS k, 'train'
  FROM read_csv('{D}/train/smp.csv', types={{'ts':'TIMESTAMPTZ'}})
  ORDER BY ts""")
log("sites, smp")

con.execute(f"CREATE TABLE generation AS SELECT site_id, {KST}, energy FROM read_parquet('{D}/train/generation.parquet') ORDER BY site_id, ts")
log("generation")

for v in ("wa", "wb"):
    con.execute(f"""CREATE TABLE forecast_{v} AS
      SELECT * EXCLUDE (ts), {KST}, 'train' AS split FROM read_parquet('{D}/train/forecast_{v}.parquet')
      UNION ALL
      SELECT * EXCLUDE (ts), {KST}, 'test' FROM read_parquet('{D}/test/forecast_{v}.parquet')
      ORDER BY {v}_station, round, ts""")
    log(f"forecast_{v}")
    con.execute(f"CREATE TABLE observed_{v} AS SELECT * EXCLUDE (ts), {KST} FROM read_parquet('{D}/train/observed_{v}.parquet') ORDER BY {v}_station, ts")
    log(f"observed_{v}")

# 베이스라인 산출물 (있을 때만)
if os.path.exists(f"{D}/composition.csv"):
    con.execute(f"CREATE TABLE baseline_composition AS SELECT * FROM read_csv('{D}/composition.csv')")
    con.execute(f"CREATE TABLE baseline_forecast AS SELECT \"group\", {KST}, forecast_kwh FROM read_csv('{D}/forecast.csv', types={{'ts':'TIMESTAMPTZ'}}) ORDER BY 1,2")
    con.execute(f"CREATE TABLE baseline_sites AS SELECT site_id, {KST}, kwh FROM read_parquet('{D}/baseline_sites.parquet') ORDER BY 1,2")
    log("baseline_*")

# 편의 뷰
con.execute("""CREATE VIEW v_smp AS
  SELECT ts, ts::DATE AS date, hour(ts) AS hour, smp, k, smp*k AS weight, split FROM smp""")
con.execute("""CREATE VIEW v_generation AS
  SELECT g.site_id, g.ts, g.ts::DATE AS date, hour(g.ts) AS hour, g.energy,
         s.capacity_kw, g.energy / s.capacity_kw AS cf, s.current_group, s.lat, s.lon, s.wa_station, s.wb_station
  FROM generation g JOIN sites s USING (site_id)""")
con.execute("""CREATE VIEW v_generation_hourly_total AS
  SELECT ts, count(*) AS n_sites, sum(energy) AS energy_kwh FROM generation GROUP BY ts ORDER BY ts""")
con.execute("""CREATE VIEW v_site_coverage AS
  SELECT s.site_id, s.capacity_kw, s.current_group, count(g.ts) AS n_rows, min(g.ts) AS first_ts, max(g.ts) AS last_ts,
         sum(g.energy) AS total_kwh
  FROM sites s LEFT JOIN generation g USING (site_id) GROUP BY ALL ORDER BY s.site_id""")
log("views")

print()
print("=== row counts ===")
for t in ["sites","smp","generation","forecast_wa","forecast_wb","observed_wa","observed_wb",
          "baseline_composition","baseline_forecast","baseline_sites"]:
    try:
        n = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        extra = ""
        if t.startswith("forecast") or t=="smp":
            extra = str(con.execute(f"SELECT split, count(*) FROM {t} GROUP BY split ORDER BY split").fetchall())
        print(f"{t:22s} {n:>12,} {extra}")
    except Exception as e: print(t, "skip", e)
print("ts sample:", con.execute("SELECT min(ts), max(ts) FROM generation").fetchone())
con.close()
print("DB size MB:", round(os.path.getsize(DB)/1e6,1))
