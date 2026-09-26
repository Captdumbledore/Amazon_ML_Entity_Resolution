#!/usr/bin/env python3
import os
import shutil
import glob
from ngram_retrieval_experiment_v2 import init_duckdb, process_country_config, CHECKPOINT_DIR

def run_test():
    print("=== STARTING RESUME TEST ===")
    
    # 1. Clean up old checkpoints for test
    test_dir = os.path.join(CHECKPOINT_DIR, "3_5", "US")
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        
    conn = init_duckdb()
    
    # 2. Run with simulated crash at batch 3
    print("\n--- PHASE A: Run until crash ---")
    try:
        process_country_config(conn, 'US', (3,5), k=10, batch_size=2000, test_crash_at_batch=3, test_mode=True)
    except SystemExit as e:
        if e.code == 99:
            print("Successfully caught simulated crash at batch 3.")
        else:
            raise
            
    # Check directory state
    files = sorted(glob.glob(os.path.join(test_dir, "*.parquet")))
    print(f"\nFiles after crash: {[os.path.basename(f) for f in files]}")
    assert "batch_000001.parquet" in [os.path.basename(f) for f in files], "Batch 1 missing"
    assert "batch_000002.parquet" in [os.path.basename(f) for f in files], "Batch 2 missing"
    assert "batch_000003.parquet" not in [os.path.basename(f) for f in files], "Batch 3 should not be complete"
    
    # 3. Resume and complete
    print("\n--- PHASE B: Resume and complete ---")
    process_country_config(conn, 'US', (3,5), k=10, batch_size=2000, test_crash_at_batch=None, test_mode=True)
    
    files_after = sorted(glob.glob(os.path.join(test_dir, "*.parquet")))
    print(f"\nFiles after resume: {[os.path.basename(f) for f in files_after]}")
    assert len(files_after) >= 5, "Should have 5 batches"
    
    print("\n=== RESUME TEST PASSED ===")

if __name__ == '__main__':
    run_test()
