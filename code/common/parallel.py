"""Run a stage's jobs on many CPU cores at once, within a memory budget, saving each finished job so a run can resume.

A stage is a list of independent JOBS (for example "restore picture 7 with every method"). Each job is a small dict with a
unique "key". `run_stage` does four things:
  1. skips every job whose key is already saved in the stage's checkpoint file (that is the resume),
  2. decides how many worker processes to use: at most 16 (never more than 20), and few enough that they fit in the memory
     budget (at most 16 GB, and never more than 75% of the memory that is free right now),
  3. runs the remaining jobs in parallel, saving each result to the checkpoint the moment it arrives,
  4. watches memory while running; if the workers use more than the budget, or the system runs short of free memory, it stops
     the pool, halves the number of workers and carries on with the unfinished jobs (nothing finished is lost).

The checkpoint file name contains a short fingerprint of the stage's settings, so changing a setting (or the config
file) starts a fresh run instead of mixing old and new results.
"""
import hashlib
import json
import multiprocessing
import os
import time
import traceback
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
CHECKPOINTS = PROJECT / "results" / "checkpoints"

DEFAULT_WORKERS = 16     # other work runs on this computer too, so we use only part of it
MAX_WORKERS = 20         # a hard ceiling: even `--workers 30` is cut down to this
MEMORY_BUDGET_GB = 16.0  # the workers together may use at most this much memory ...
MEMORY_SHARE_OF_FREE = 0.75   # ... and never more than this share of what is free when the stage starts
LOW_FREE_MEMORY_GB = 2.0      # if the whole system drops below this much free memory, back off


def fingerprint(*things):
    """A short code (8 characters) that changes whenever any of `things` (settings, config text) changes."""
    return hashlib.sha1(json.dumps(things, sort_keys=True, default=str).encode()).hexdigest()[:8]


def free_memory_gb():
    """Memory the system could give a new program right now, in GB (the 'available' line of /proc/meminfo)."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 ** 2  # the file counts in kB
    return float("inf")


def workers_memory_gb():
    """Memory used right now by all worker processes together, in GB (read from /proc, no extra library needed)."""
    total_kb = 0
    for child in multiprocessing.active_children():
        try:
            for line in Path(f"/proc/{child.pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total_kb += int(line.split()[1])
        except OSError:  # the worker just ended
            pass
    return total_kb / 1024 ** 2


def choose_workers(requested, memory_per_worker_gb):
    """How many workers to start: the request, capped at 20, and few enough that their expected memory fits the budget."""
    budget = min(MEMORY_BUDGET_GB, MEMORY_SHARE_OF_FREE * free_memory_gb())
    by_memory = max(1, int(budget / memory_per_worker_gb))
    by_cpu = max(1, min(requested or DEFAULT_WORKERS, MAX_WORKERS, (os.cpu_count() or 2) - 2))
    print(f"  workers: {min(by_cpu, by_memory)} (CPU allows {by_cpu}, memory budget {budget:.1f} GB / {memory_per_worker_gb} GB each allows {by_memory})", flush=True)
    return min(by_cpu, by_memory), budget


def _run_one(packed):
    """Worker side: run one job and return (key, rows or None, error text or None). Never raises, so one bad job cannot stop the pool."""
    work, job = packed
    try:
        return job["key"], work(job), None
    except Exception:  # noqa: BLE001 - we want to report any failure and carry on
        return job["key"], None, traceback.format_exc(limit=4)


def _quiet_threads():
    """Each worker should use one thread, otherwise many workers x many math threads fight for the same cores."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"


def run_stage(stage_name, signature, jobs, work, workers=None, memory_per_worker_gb=1.0, resume=True):
    """Run `work(job)` for every job in parallel and return all result rows (a list of dicts), resumed from the checkpoint.

    stage_name            short name used for the checkpoint file, e.g. "haze_matched"
    signature             fingerprint of the settings (see `fingerprint`)
    jobs                  list of dicts, each with a unique "key"
    work                  function taking one job and returning a list of result rows; it must be defined at the top of a module
    memory_per_worker_gb  rough memory one worker needs (used to size the pool; the live check corrects a wrong guess)
    """
    keys = [job["key"] for job in jobs]
    if len(set(keys)) != len(keys):  # two jobs sharing a key would overwrite each other's saved results
        repeated = sorted({k for k in keys if keys.count(k) > 1})[:3]
        raise ValueError(f"[{stage_name}] job keys must be unique; repeated: {repeated}")
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINTS / f"{stage_name}_{signature}.jsonl"
    if not resume and checkpoint.exists():
        checkpoint.unlink()

    # --- 1. read what is already finished
    finished = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                finished[record["key"]] = record["rows"]
    print(f"[{stage_name}] {len(jobs)} jobs: {len(finished)} already done, {len(jobs) - len(finished)} to run", flush=True)

    # --- 2 and 3. run what is left; if memory runs short, stop, halve the workers and go on
    _quiet_threads()
    failures = []
    pool_size, budget = choose_workers(workers, memory_per_worker_gb)
    started, done = time.time(), 0
    while True:
        todo = [job for job in jobs if job["key"] not in finished and job["key"] not in dict(failures)]
        if not todo:
            break
        backed_off = False
        with multiprocessing.get_context("fork").Pool(min(pool_size, len(todo))) as pool, open(checkpoint, "a") as saved:
            for key, rows, error in pool.imap_unordered(_run_one, [(work, job) for job in todo]):
                done += 1
                if error:
                    failures.append((key, error))
                    print(f"[{stage_name}] job {key} FAILED:\n{error}", flush=True)
                else:
                    saved.write(json.dumps({"key": key, "rows": rows}) + "\n")
                    saved.flush()
                    finished[key] = rows
                # --- 4. watch memory; step down if the workers use more than the budget or the system runs low
                used, free = workers_memory_gb(), free_memory_gb()
                if (used > budget or free < LOW_FREE_MEMORY_GB) and pool_size > 1:
                    pool_size = max(1, pool_size // 2)
                    print(f"[{stage_name}] memory pressure (workers use {used:.1f} GB, system has {free:.1f} GB free): "
                          f"stopping the pool and continuing with {pool_size} workers", flush=True)
                    backed_off = True
                    pool.terminate()
                    break
                if done % max(1, len(jobs) // 10) == 0:
                    elapsed = time.time() - started
                    left = len(jobs) - len(finished) - len(failures)
                    print(f"[{stage_name}] {len(finished)}/{len(jobs)} done, {elapsed:.0f}s elapsed, "
                          f"memory {used:.1f} GB in workers, {free:.1f} GB free, about {elapsed / done * left:.0f}s left", flush=True)
        if not backed_off:
            break

    if failures:
        raise RuntimeError(f"[{stage_name}] {len(failures)} job(s) failed; fix the cause and run again to resume (finished jobs are kept)")

    # --- return every row in the original job order
    return [row for job in jobs for row in finished[job["key"]]]
