# Phase 3 Resume Test Report

## 1. Objective
Verify that the redesigned chunk-level checkpointing system safely recovers from mid-batch interruptions without data corruption, duplication, or redundant recomputation.

## 2. Test Design (`src/test_resume.py`)
- Clean any existing checkpoints for the test configuration.
- Launch Phase A (simulated interruption): Process batches 1 and 2 successfully, but forcefully trigger `sys.exit(99)` in the middle of processing batch 3, right before the `.tmp` file can be atomically renamed.
- Verify directory state after crash.
- Launch Phase B (resume): Relaunch the exact same retrieval process with no forced crashes.
- Verify final directory state.

## 3. Results
**Phase A (Crash):**
The script successfully vectorized 20,000 targets and 10,000 S1 validation queries. It processed and cleanly finalized `batch_000001.parquet` and `batch_000002.parquet`. As expected, the simulated crash killed the process during batch 3. Inspection of the directory confirmed that only batches 1 and 2 existed.

**Phase B (Resume):**
Upon restart, the script correctly detected that `batch_000001.parquet` and `batch_000002.parquet` were complete and skipped them. It successfully resumed execution at batch 3, completing batches 3, 4, and 5.

## 4. Conclusion
The resume test **PASSED**. 
- Incomplete/interrupted batches are never finalized because of the atomic rename `batch.tmp -> batch.parquet`.
- Completed batches are safely bypassed.
- Rerunning the script after an interruption seamlessly picks up from the last finalized checkpoint.

The system is fully safe to run on the full 441,365-entity validation split.
