import duckdb
import time

conn = duckdb.connect()

# Set memory limit to stay well within available RAM (e.g. 3GB)
conn.execute("SET max_memory = '3GB'")
conn.execute("SET preserve_insertion_order = false")

print("Checking most frequent raw and normalized names in S2 and S3...")
t0 = time.time()

# Let's inspect frequency of names in S2 + S3
query = """
WITH targets AS (
    SELECT business_name, country FROM read_csv('Dataset/student_resource/dataset/train/train_source2.tsv', delim='\t', header=true)
    UNION ALL
    SELECT business_name, country FROM read_csv('Dataset/student_resource/dataset/train/train_source3.tsv', delim='\t', header=true)
)
SELECT business_name, count(*) as cnt
FROM targets
GROUP BY business_name
ORDER BY cnt DESC
LIMIT 25;
"""
res = conn.execute(query).fetchall()
print(f"Top 25 most frequent names in S2+S3 (in {time.time()-t0:.2f}s):")
for name, cnt in res:
    print(f"  {cnt:6d} : {name}")
