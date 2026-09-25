import duckdb

conn = duckdb.connect()
conn.execute("SET max_memory = '4GB'")
conn.execute("SET threads = 8")

q = """
WITH name_pairs AS (
    SELECT s1.entity_id as s1_id, t.entity_id as target_id
    FROM 'data/train_s1.parquet' s1
    JOIN (
        SELECT entity_id, norm_name, country FROM 'data/train_s2.parquet'
        UNION ALL
        SELECT entity_id, norm_name, country FROM 'data/train_s3.parquet'
    ) t ON s1.country = t.country AND s1.norm_name = t.norm_name
    WHERE s1.is_val = true AND s1.norm_name != ''
),
addr_pairs AS (
    SELECT s1.entity_id as s1_id, t.entity_id as target_id
    FROM 'data/train_s1.parquet' s1
    JOIN (
        SELECT entity_id, norm_addr, country FROM 'data/train_s2.parquet'
        UNION ALL
        SELECT entity_id, norm_addr, country FROM 'data/train_s3.parquet'
    ) t ON s1.country = t.country AND s1.norm_addr = t.norm_addr
    WHERE s1.is_val = true AND s1.norm_addr != ''
),
union_pairs AS (
    SELECT s1_id, target_id FROM name_pairs
    UNION
    SELECT s1_id, target_id FROM addr_pairs
),
labeled AS (
    SELECT u.s1_id, u.target_id, (g.target_entity_id IS NOT NULL) as is_tp
    FROM union_pairs u
    LEFT JOIN 'data/train_gt_pairs.parquet' g
      ON u.s1_id = g.source1_entity_id AND u.target_id = g.target_entity_id
)
SELECT 
    count(*) as total_candidates,
    sum(case when is_tp then 1 else 0 end) as tp,
    sum(case when not is_tp then 1 else 0 end) as fp
FROM labeled;
"""
res = conn.execute(q).fetchone()
tot = res[0]
tp = res[1]
fp = res[2]
val_gt = 1527614

print(f"Union (Country + Exact Norm Name OR Exact Norm Addr):")
print(f"  Total Candidates: {tot:,}")
print(f"  True Positives  : {tp:,} (Recall: {tp/val_gt*100:.2f}%)")
print(f"  False Positives : {fp:,} (Precision: {tp/tot*100:.2f}%)")
