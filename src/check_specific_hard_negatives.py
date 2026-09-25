import duckdb

conn = duckdb.connect()
conn.execute("SET max_memory = '3GB'")

names = [
    'Pediatric Medicine PLLC',
    'Lakshmi Consultants Private Limited',
    'Ss Technology Private Limited',
    'Heritage Society LLC',
    'Slate PLLC',
    'Apex Center'
]

print("Investigating specific hard negative names across S1, S2, S3...")
for n in names:
    s1_cnt = conn.execute(f"SELECT count(*) FROM read_csv('Dataset/student_resource/dataset/train/train_source1.tsv', delim='\t', header=true) WHERE lower(business_name) = lower('{n}')").fetchone()[0]
    s2_cnt = conn.execute(f"SELECT count(*) FROM read_csv('Dataset/student_resource/dataset/train/train_source2.tsv', delim='\t', header=true) WHERE lower(business_name) = lower('{n}')").fetchone()[0]
    s3_cnt = conn.execute(f"SELECT count(*) FROM read_csv('Dataset/student_resource/dataset/train/train_source3.tsv', delim='\t', header=true) WHERE lower(business_name) = lower('{n}')").fetchone()[0]
    print(f"Name: '{n}': S1={s1_cnt}, S2={s2_cnt}, S3={s3_cnt}, Total S2+S3={s2_cnt+s3_cnt}, Potential pairs={s1_cnt*(s2_cnt+s3_cnt)}")
