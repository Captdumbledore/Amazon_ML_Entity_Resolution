import duckdb

conn = duckdb.connect()
res_val = conn.execute("SELECT count(*), sum(match_count) FROM 'data/train_gt_s1.parquet' WHERE is_val = true").fetchone()
res_all = conn.execute("SELECT count(*), sum(match_count) FROM 'data/train_gt_s1.parquet'").fetchone()

print(f"Validation S1: {res_val[0]:,}, GT pairs: {res_val[1]:,}")
print(f"Full Train S1: {res_all[0]:,}, GT pairs: {res_all[1]:,}")
print(f"Fraction GT pairs in Val: {res_val[1]/res_all[1]*100:.2f}%")
