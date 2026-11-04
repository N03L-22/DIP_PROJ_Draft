"""Run the experiment stages one after another, each in parallel inside, each resumable.

  matched    every method on the synthetic matched pairs, all four media          (Stage 4)
  real       every method on the public photographs (haze, glass, water)          (Stage 4)
  sweeps     one setting at a time, to see how each parameter changes the result  (Stage 4b)
  severity   how each method copes as the degradation grows                       (Stage 4c)
  pipelines  every ordered sequence of steps: does the order matter, can any beat DCP?   (Stage 4d)
  compare    do synthetic and real pictures rank the methods alike?                (Stage 5)
  failure    why methods fail: dark channel, transmission accuracy, colour cast    (Stage 6a)
  stacked    pictures degraded by two or three media at once                       (Stage 6b)
  local      the same method applied globally against tile by tile / region by region   (Stage 4e)

A stage starts only when the one before it has finished. When a stage finishes it is recorded in
results/checkpoints/stages_done.json, so running this again skips it. If a run is interrupted in the middle of a stage,
running it again continues that stage from the jobs already saved, not from the start.

  --only matched,sweeps     run just these stages          --from sweeps    start at this stage
  --force sweeps            run this stage again from the saved jobs (use --fresh to discard them too)
  --workers 12              parallel processes (default 16, never more than 20)
  --quick                   a small, fast version of everything, to check the pipeline works

Run from the project root:  .venv/bin/python code/experiments/run_stages.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import (compare_real_vs_synthetic, run_experiments, run_failure_analysis, run_pipelines, run_severity_sweep,  # noqa: E402
                         run_local_vs_global, run_stacked, run_sweeps)

PROJECT = Path(__file__).resolve().parents[2]
STATUS_FILE = PROJECT / "results" / "checkpoints" / "stages_done.json"
MEDIA = ("haze", "water", "glass", "plastic")


def stage_matched(options):
    for medium in MEDIA:
        run_experiments.run(medium, "matched", options["per_level"], options["workers"], options["resume"])


def stage_real(options):
    for medium in ("haze", "glass", "water"):  # water has no reference: only the reference-free scores are filled in
        run_experiments.run(medium, "real", options["per_level"], options["workers"], options["resume"])


def stage_sweeps(options):
    for medium in MEDIA:
        run_sweeps.run(medium, "medium", options["sweep_pictures"], 320, options["workers"], options["resume"])


def stage_severity(options):
    for medium in MEDIA:
        run_severity_sweep.run(medium, options["sweep_pictures"], 320, options["workers"], options["resume"])


def stage_pipelines(options):
    for medium in MEDIA:
        run_pipelines.run(medium, options["workers"], options["resume"], options["quick"])


def stage_compare(options):
    compare_real_vs_synthetic.main()


def stage_failure(options):
    run_failure_analysis.run(options["workers"], options["resume"])


def stage_stacked(options):
    run_stacked.run(options["workers"], options["resume"])


def stage_local(options):
    run_local_vs_global.run(options["workers"], options["resume"])


STAGES = {"matched": stage_matched, "real": stage_real, "sweeps": stage_sweeps, "severity": stage_severity, "pipelines": stage_pipelines,
          "compare": stage_compare, "failure": stage_failure, "stacked": stage_stacked, "local": stage_local}


def read_status():
    return json.loads(STATUS_FILE.read_text()) if STATUS_FILE.exists() else {}


def write_status(status):
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="comma-separated stage names to run")
    parser.add_argument("--from", dest="start", choices=list(STAGES), help="start at this stage")
    parser.add_argument("--force", help="comma-separated stage names to run again even if recorded as done")
    parser.add_argument("--fresh", action="store_true", help="also discard the saved jobs of the stages that run")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="small and fast: 3 pictures per level, 3 pictures per sweep")
    arguments = parser.parse_args()

    options = {"per_level": 3 if arguments.quick else 12, "sweep_pictures": 3 if arguments.quick else 6,
               "workers": arguments.workers, "resume": not arguments.fresh, "quick": arguments.quick}
    names = list(STAGES)
    if arguments.only:
        names = [n for n in arguments.only.split(",") if n in STAGES]
    if arguments.start:
        names = [n for n in names if list(STAGES).index(n) >= list(STAGES).index(arguments.start)]
    forced = set(arguments.force.split(",")) if arguments.force else set()
    if arguments.quick:  # a quick run is a different experiment; never mix it up with the real record
        STATUS_FILE = STATUS_FILE.with_name("stages_done_quick.json")

    status = read_status()
    overall = time.time()
    for name in names:
        if name in status and name not in forced and not arguments.fresh:
            print(f"=== stage '{name}' already done ({status[name]['seconds']:.0f}s), skipping (use --force {name} to repeat)", flush=True)
            continue
        print(f"\n=== stage '{name}' ===", flush=True)
        started = time.time()
        STAGES[name](options)  # stops here with an error if any job failed; the stage is then NOT recorded as done
        status[name] = {"seconds": round(time.time() - started, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        write_status(status)
        print(f"=== stage '{name}' finished in {status[name]['seconds']:.0f}s", flush=True)
    print(f"\nall requested stages finished; total {time.time() - overall:.0f}s")
