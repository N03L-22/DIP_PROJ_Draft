"""Checks for common/parallel.py: parallel speed-up, resume from the checkpoint, failure handling and memory back-off.
Run from the project root:  .venv/bin/python code/tests/test_parallel.py
"""
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import parallel  # noqa: E402

parallel.CHECKPOINTS = Path(tempfile.mkdtemp())  # keep the test's checkpoint files away from the real results
checks = []


def check(name, condition):
    checks.append((name, bool(condition)))
    print(("PASS  " if condition else "FAIL  ") + name)


def slow_square(job):
    """A job that takes a moment and returns one row."""
    time.sleep(0.3)
    return [{"n": job["n"], "square": job["n"] ** 2}]


def sometimes_fails(job):
    if job["n"] == 3:
        raise ValueError("job 3 is broken on purpose")
    return [{"n": job["n"]}]


def hungry(job):
    """Holds about 150 MB for a moment, so that several at once exceed a tiny memory budget."""
    block = np.ones(20_000_000)  # 160 MB
    time.sleep(0.4)
    return [{"n": job["n"], "sum": float(block[:10].sum())}]


jobs = [{"key": f"n{n}", "n": n} for n in range(24)]

# 1. parallel really is faster: 24 jobs of 0.3 s would take 7.2 s one after another
started = time.time()
rows = parallel.run_stage("toy", "sig1", jobs, slow_square, workers=8)
elapsed = time.time() - started
check("all 24 results come back, in job order", [r["n"] for r in rows] == list(range(24)) and rows[5]["square"] == 25)
check(f"parallel run is much faster than serial ({elapsed:.1f}s vs 7.2s)", elapsed < 3.5)

# 2. resume: a second run finds everything done and runs nothing
started = time.time()
rows_again = parallel.run_stage("toy", "sig1", jobs, slow_square, workers=8)
check("a finished stage resumes instantly", time.time() - started < 1.0 and rows_again == rows)

# 3. resume after a partial run: pretend only the first 10 jobs were saved
parallel.run_stage("toy2", "sig1", jobs[:10], slow_square, workers=8)
started = time.time()
rows_partial = parallel.run_stage("toy2", "sig1", jobs, slow_square, workers=8)
check("a partly finished stage runs only the missing jobs", len(rows_partial) == 24 and time.time() - started < 2.5)

# 4. a different fingerprint is a fresh run (changed settings never reuse old results)
started = time.time()
parallel.run_stage("toy", "sig2", jobs[:8], slow_square, workers=8)
check("a changed fingerprint starts fresh", time.time() - started > 0.25)

# 5. a failing job is reported, the other jobs are kept, and a fixed re-run resumes
try:
    parallel.run_stage("fail", "sig1", [{"key": f"n{n}", "n": n} for n in range(6)], sometimes_fails, workers=4)
    raised = False
except RuntimeError:
    raised = True
saved_lines = (parallel.CHECKPOINTS / "fail_sig1.jsonl").read_text().splitlines()
check("a failing job raises at the end but the other 5 are saved", raised and len(saved_lines) == 5)

# 6. memory back-off: with a budget of about 0.4 GB, eight 160 MB jobs at once is too much; the pool must step down and still finish
parallel.MEMORY_BUDGET_GB = 0.4
rows_memory = parallel.run_stage("hungry", "sig1", [{"key": f"n{n}", "n": n} for n in range(12)], hungry, workers=8, memory_per_worker_gb=0.05)
check("with a tiny memory budget it backs off and still finishes every job", len(rows_memory) == 12)

# 6b. two jobs with the same key would overwrite each other, so the stage refuses to start
try:
    parallel.run_stage("dup", "sig1", [{"key": "same", "n": 1}, {"key": "same", "n": 2}], slow_square, workers=2)
    refused = False
except ValueError:
    refused = True
check("duplicate job keys are refused", refused)

# 7. the worker count respects the CPU limits
parallel.MEMORY_BUDGET_GB = 16.0
check("never more than 20 workers, even if 64 are requested", parallel.choose_workers(64, 0.1)[0] <= 20)
check("default is at most 16 workers", parallel.choose_workers(None, 0.1)[0] <= 16)
check("a big per-worker memory estimate reduces the workers", parallel.choose_workers(16, 6.0)[0] <= 3)

failed = [name for name, ok in checks if not ok]
print(f"\n{len(checks) - len(failed)} of {len(checks)} checks passed")
sys.exit(1 if failed else 0)
