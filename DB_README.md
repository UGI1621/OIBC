# oibc.duckdb — OIBC 2026 데이터 조회용 DB

`OIBC_2026_DATA/handover 2/01_participant_dataset/`의 CSV·parquet 전체를 DuckDB 단일 파일로 적재한 것.
원본은 수정하지 않았고, 이 파일은 `python build_db.py`로 언제든 재생성 가능(약 10초).

## 보는 방법

| 방법 | 명령 |
|---|---|
| 테이블 목록 + 행 수 | `python query.py` |
| SQL 한 줄 실행 | `python query.py "SELECT * FROM v_smp WHERE k=3 LIMIT 20"` |
| 결과를 CSV로 저장 | `python query.py "SELECT ..." --csv out.csv` |
| 브라우저 SQL UI | `python query.py --ui` → http://localhost:4213 (테이블 탐색·쿼리·결과 표) |
| Python에서 직접 | `import duckdb; con = duckdb.connect("oibc.duckdb", read_only=True); con.sql("...").df()` |

주의: DuckDB는 쓰기 모드로 연 프로세스가 있으면 다른 프로세스가 파일을 열 수 없다. `query.py`는 모두 읽기 전용으로 열므로 UI와 스크립트를 동시에 써도 되지만, `build_db.py`(재생성)를 돌릴 때는 UI를 먼저 닫아야 한다.

VS Code 안에서 보려면 확장 **DuckDB SQL Tools** (RandomFractalsInc) 또는 **SQLTools + DuckDB Driver**를 설치해 `oibc.duckdb`에 연결.
설치돼 있는 Database Client(cweijan)는 DuckDB를 지원하지 않음.

## 시각 규약

- `ts`는 **KST 기준 naive TIMESTAMP** (원본의 `+09:00`을 떼어낸 값). 우측닫힘: `13:00`은 12:00~13:00 구간.
- 날짜·시 필터는 `ts::DATE`, `hour(ts)` 또는 뷰의 `date`, `hour` 컬럼 사용.

## 테이블

| 테이블 | 행 | 키 | 비고 |
|---|---|---|---|
| `sites` | 253 | `site_id` | 용량·좌표·관측지점·거리 |
| `smp` | 47,448 | `ts` | `split`='test'(8,760, 채점 시각) / 'train'(38,688). `k`는 train에도 규칙(11~15시=3)으로 생성 |
| `generation` | 4,392,566 | `site_id, ts` | 학습 구간 발전량(kWh). 결측은 행 부재 |
| `forecast_wa` | 5,778,812 | `wa_station, round, ts` | 제공사 A 예보. `split` train/test. round 10/17 |
| `forecast_wb` | 6,900,956 | `wb_station, round, ts` | 제공사 B 예보. 풍속 km/h |
| `observed_wa` | 2,359,974 | `wa_station, ts` | A 실측(학습 구간만) |
| `observed_wb` | 2,824,223 | `wb_station, ts` | B 실측(학습 구간만) |
| `baseline_composition` | 253 | `site_id` | baseline.py 산출 |
| `baseline_forecast` | 17,520 | `group, ts` | baseline.py 산출 |
| `baseline_sites` | 2,216,280 | `site_id, ts` | baseline.py 발전소별 예측 |

## 뷰

| 뷰 | 내용 |
|---|---|
| `v_smp` | `smp` + `date`, `hour`, `weight`(=smp×k) |
| `v_generation` | `generation` ⋈ `sites`: `cf`(이용률=energy/capacity), 그룹, 좌표, 관측지점 |
| `v_generation_hourly_total` | 시각별 전체 발전량 합과 데이터 있는 발전소 수 |
| `v_site_coverage` | 발전소별 행 수, 첫/마지막 시각, 누적 발전량 |

## 예시 쿼리

```sql
-- 채점 가중치가 큰 날 TOP 10
SELECT date, sum(weight) w FROM v_smp WHERE split='test' GROUP BY date ORDER BY w DESC LIMIT 10;

-- 발전소 PV004의 2025-05 정오 이용률
SELECT ts, energy, cf FROM v_generation WHERE site_id='PV004' AND hour=13 AND date BETWEEN '2025-05-01' AND '2025-05-31';

-- 발전소 하나의 예보(A, round 17)와 실적을 시각으로 붙이기
SELECT g.ts, g.energy, f.wa_cloud, f.wa_temp, f.wa_rain
FROM v_generation g
JOIN forecast_wa f ON f.wa_station = g.wa_station AND f.ts = g.ts AND f.round = 17 AND f.split = 'train'
WHERE g.site_id = 'PV001' AND g.date = '2025-05-30' ORDER BY g.ts;

-- 관측지점별 예보 vs 실측 운량 오차 (A)
SELECT f.wa_station, round(avg(abs(f.wa_cloud - o.wa_cloud)),1) mae_cloud
FROM forecast_wa f JOIN observed_wa o USING (wa_station, ts)
WHERE f.round = 17 AND f.split = 'train' GROUP BY 1 ORDER BY 2 DESC;

-- 데이터가 1년 미만인 신규 발전소
SELECT * FROM v_site_coverage WHERE first_ts > '2024-06-01' ORDER BY first_ts;
```
