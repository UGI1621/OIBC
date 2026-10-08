"""oibc.duckdb 조회 헬퍼.

    python query.py                         # 테이블 목록과 행 수
    python query.py "SELECT * FROM v_smp WHERE k=3 LIMIT 20"
    python query.py "SELECT ..." --csv out.csv   # 결과를 CSV로 저장
    python query.py --ui                    # 브라우저 SQL UI (localhost:4213) 실행
"""
import sys, os, duckdb, pandas as pd

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oibc.duckdb")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)

def main():
    args = [a for a in sys.argv[1:]]
    if "--ui" in args:
        con = duckdb.connect(DB, read_only=True)   # 읽기 전용: UI를 켜 둔 채 다른 스크립트도 DB를 읽을 수 있음
        con.execute("INSTALL ui; LOAD ui; CALL start_ui();")
        print("DuckDB UI: http://localhost:4213  (종료: Ctrl+C)")
        try:
            import time
            while True: time.sleep(3600)
        except KeyboardInterrupt:
            return
    con = duckdb.connect(DB, read_only=True)
    if not args:
        rows = con.execute("SELECT table_name, table_type FROM information_schema.tables ORDER BY table_type, table_name").fetchall()
        for name, typ in rows:
            n = con.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
            print(f"{typ:10s} {name:28s} {n:>12,}")
        return
    out = None
    if "--csv" in args:
        i = args.index("--csv"); out = args[i + 1]; del args[i:i + 2]
    sql = " ".join(args)
    df = con.execute(sql).df()
    if out:
        df.to_csv(out, index=False); print(f"{len(df):,} rows -> {out}")
    else:
        print(df.to_string(index=False)); print(f"({len(df):,} rows)")

if __name__ == "__main__":
    main()
