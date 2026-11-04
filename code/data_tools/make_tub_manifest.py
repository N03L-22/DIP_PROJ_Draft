"""Turn the TUB tank dataset into a paired water manifest: each turbid picture with the lowest-turbidity picture of the same view.

TUB (Aalborg University, CC-BY-4.0): 33 static scenes, 4 cameras, 10 turbidity stages made by adding measured oat milk to a tank. Scene, camera and
objects stay fixed, so Stage01 of an experiment and camera is the same view as its later stages, and is used as the reference. Stage01 is the LEAST
turbid picture, not a perfectly clear one (its turbidity is listed in av_turbidity.csv), so scores are against a nearly clear reference.
Each pair is labelled by the stage's average turbidity in NTU: mild below 10, medium 10 to 20, heavy 20 and above (the dataset's own low / medium / high).

Output:  data/curated/manifest_water_tub.csv   (same columns as the other curated manifests; paths relative to data/curated)
Run from the project root:  .venv/bin/python code/data_tools/make_tub_manifest.py
"""
import csv
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
TUB = PROJECT / "data" / "existing" / "TUB"
OUT = PROJECT / "data" / "curated" / "manifest_water_tub.csv"

with open(TUB / "av_turbidity.csv") as handle:
    ntu = {(r["Exp_id"], int(r["stage"])): float(r["avg"]) for r in csv.DictReader(handle)}

rows = []
for first in sorted((TUB / "images").glob("*_Stage01.jpg")):
    experiment, camera = first.name.split("_")[:2]
    for stage in range(2, 11):
        degraded = TUB / "images" / f"{experiment}_{camera}_Stage{stage:02d}.jpg"
        if not degraded.exists() or (experiment, stage) not in ntu:
            continue
        level = "mild" if ntu[(experiment, stage)] < 10 else "medium" if ntu[(experiment, stage)] < 20 else "heavy"
        rows.append({"medium": "water", "source": "TUB", "id": f"{experiment}_{camera}_S{stage:02d}", "level": level, "ntu": ntu[(experiment, stage)],
                     "reference_ntu": ntu.get((experiment, 1), ""), "degraded_path": Path("../existing/TUB/images") / degraded.name,
                     "reference_path": Path("../existing/TUB/images") / first.name})
with open(OUT, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
counts = {level: sum(r["level"] == level for r in rows) for level in ("mild", "medium", "heavy")}
print(f"wrote {len(rows)} pairs to {OUT.relative_to(PROJECT)}: {counts}")
