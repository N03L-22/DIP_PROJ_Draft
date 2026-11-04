"""Pick the working set from the raw downloads and put it in data/curated/ (the raw files are never changed).

 haze   O-HAZE, I-HAZE, Dense-Haze, NH-Haze: every pair, sorted into mild / medium / heavy by measured fog
        RESIDE SOTS: 100 evenly spaced synthetic pairs (a check on the haze model)
 glass  raindrop pairs: every pair, sorted by how much of the picture the drops change
 water  UIEB: 150 images spread evenly across the degradation range (read straight out of the password-protected RAR)

Each pick is a hard link (no extra disk space) or an extracted file, listed in data/curated/manifest_<medium>.csv
with its source and score, and shown on a contact sheet in data/curated/contact_sheets/.
Run from the project root:  .venv/bin/python code/data_tools/select_curated.py
"""
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.rar_reader import iter_rar_files  # noqa: E402
from common.stats import colour_cast, contrast, dark_channel_mean  # noqa: E402
from data_tools.curation import CURATED, EXISTING, contact_sheet, link_or_copy, tertile_levels, write_manifest  # noqa: E402

SCORE_SIDE = 400  # pictures are shrunk to this longest side before scoring (fast, and scores barely change)


def small_float(picture):
    """A PIL image as a small float RGB array for scoring."""
    picture = picture.convert("RGB")
    picture.thumbnail((SCORE_SIDE, SCORE_SIDE))
    return np.asarray(picture, dtype=np.float64) / 255.0


# ---------------------------------------------------------------- haze (paired, real)
HAZE_SETS = {
    # name: (folder with the hazy pictures, folder with the matching clear pictures, how to turn a hazy name into a clear name)
    "O-HAZE": ("O-HAZE/# O-HAZY NTIRE 2018/hazy", "O-HAZE/# O-HAZY NTIRE 2018/GT", lambda n: n.replace("_hazy", "_GT")),
    "I-HAZE": ("I-HAZE/# I-HAZY NTIRE 2018/hazy", "I-HAZE/# I-HAZY NTIRE 2018/GT", lambda n: n.replace("_hazy", "_GT")),
    "Dense-Haze": ("Dense-Haze/hazy", "Dense-Haze/GT", lambda n: n.replace("_hazy", "_GT")),
    "NH-Haze": ("NH-Haze/NH-HAZE", "NH-Haze/NH-HAZE", lambda n: n.replace("_hazy", "_GT")),
}


def curate_real_haze():
    rows = []
    for name, (hazy_dir, clear_dir, to_clear) in HAZE_SETS.items():
        for hazy in sorted((EXISTING / hazy_dir).glob("*")):
            if "_hazy" not in hazy.name:
                continue
            clear = EXISTING / clear_dir / to_clear(hazy.name)
            if not clear.exists():
                print("no clear partner for", hazy.name)
                continue
            score = dark_channel_mean(small_float(Image.open(hazy)))
            rows.append({"medium": "haze", "source": name, "id": hazy.name.split("_")[0], "hazy": hazy, "clear": clear, "score": score})
    levels = tertile_levels([r["score"] for r in rows])
    manifest, sheet = [], []
    for row, level in zip(rows, levels):
        stem = f"{row['source']}_{row['id']}"
        hazy_out = CURATED / "haze" / level / f"{stem}_hazy{row['hazy'].suffix.lower()}"
        clear_out = CURATED / "haze" / level / f"{stem}_clear{row['clear'].suffix.lower()}"
        link_or_copy(row["hazy"], hazy_out)
        link_or_copy(row["clear"], clear_out)
        manifest.append({"medium": "haze", "source": row["source"], "id": row["id"], "level": level, "score_dark_channel": f"{row['score']:.4f}",
                         "degraded_path": hazy_out.relative_to(CURATED), "reference_path": clear_out.relative_to(CURATED),
                         "origin": str(row["hazy"].relative_to(EXISTING))})
        sheet.append((row["hazy"], f"{level[:3]} {stem} {row['score']:.2f}", row["score"]))
    write_manifest("haze", manifest)
    sheet.sort(key=lambda s: s[2])
    contact_sheet("haze", [(p, c) for p, c, _ in sheet])


# ---------------------------------------------------------------- RESIDE SOTS (synthetic pairs, a model check)
def curate_sots(per_folder=50):
    manifest, sheet = [], []
    for scene in ("outdoor", "indoor"):
        base = EXISTING / "RESIDE-SOTS" / "SOTS" / scene
        pairs = []
        for hazy in sorted((base / "hazy").glob("*")):
            clear = next(iter(sorted((base / "gt").glob(hazy.name.split("_")[0] + ".*"))), None)
            if clear is not None:
                pairs.append((hazy, clear))
        picks = np.linspace(0, len(pairs) - 1, per_folder).round().astype(int)
        chosen = [pairs[i] for i in dict.fromkeys(picks)]
        scores = [dark_channel_mean(small_float(Image.open(h))) for h, _ in chosen]
        for (hazy, clear), score, level in zip(chosen, scores, tertile_levels(scores)):
            hazy_out = CURATED / "haze_synthetic" / scene / level / hazy.name
            clear_out = CURATED / "haze_synthetic" / scene / level / f"{hazy.name.split('_')[0]}_clear{clear.suffix}"
            link_or_copy(hazy, hazy_out)
            link_or_copy(clear, clear_out)
            manifest.append({"medium": "haze_synthetic", "source": f"SOTS-{scene}", "id": hazy.stem, "level": level, "score_dark_channel": f"{score:.4f}",
                             "degraded_path": hazy_out.relative_to(CURATED), "reference_path": clear_out.relative_to(CURATED), "origin": str(hazy.relative_to(EXISTING))})
            sheet.append((hazy, f"{scene[:3]} {hazy.stem} {score:.2f}", score))
    write_manifest("haze_synthetic", manifest)
    sheet.sort(key=lambda s: s[2])
    contact_sheet("haze_synthetic", [(p, c) for p, c, _ in sheet], columns=10)


# ---------------------------------------------------------------- glass: raindrop pairs
def curate_raindrops():
    base = EXISTING / "RainDrop-pairs" / "test_a"
    rows = []
    for rainy in sorted((base / "data").glob("*_rain.png")):
        clean = base / "gt" / rainy.name.replace("_rain", "_clean")
        if not clean.exists():
            continue
        a, b = small_float(Image.open(rainy)), small_float(Image.open(clean))
        rows.append((rainy, clean, float(np.abs(a - b).mean())))  # how much the drops change the picture
    levels = tertile_levels([r[2] for r in rows])
    manifest, sheet = [], []
    for (rainy, clean, score), level in zip(rows, levels):
        rain_out = CURATED / "glass" / level / f"raindrop_{rainy.stem.split('_')[0]}_rain.png"
        clean_out = CURATED / "glass" / level / f"raindrop_{rainy.stem.split('_')[0]}_clean.png"
        link_or_copy(rainy, rain_out)
        link_or_copy(clean, clean_out)
        manifest.append({"medium": "glass", "source": "RainDrop-test_a", "id": rainy.stem.split("_")[0], "level": level, "score_mean_abs_diff": f"{score:.4f}",
                         "degraded_path": rain_out.relative_to(CURATED), "reference_path": clean_out.relative_to(CURATED), "origin": str(rainy.relative_to(EXISTING))})
        sheet.append((rainy, f"{level[:3]} {rainy.stem.split('_')[0]} {score:.3f}", score))
    write_manifest("glass", manifest)
    sheet.sort(key=lambda s: s[2])
    contact_sheet("glass_raindrops", [(p, c) for p, c, _ in sheet])


# ---------------------------------------------------------------- water: UIEB read straight from the RAR
def rank_average(values_low_is_bad, values_high_is_bad):
    """Combine several measures into one 0..1 degradation score by averaging their ranks (no unit problems)."""
    def ranks(x):
        order = np.argsort(np.argsort(x))
        return order / max(len(x) - 1, 1)
    parts = [ranks(-np.asarray(v)) for v in values_low_is_bad] + [ranks(np.asarray(v)) for v in values_high_is_bad]
    return np.mean(parts, axis=0)


UIEB_PASSWORD = "1234567"  # printed next to the raw-images link on the UIEB project page


def curate_water(how_many=150):
    archive = EXISTING / "UIEB" / "raw-890.rar"
    if not archive.exists():
        print("UIEB archive not found, skipping water")
        return
    names, stats = [], []
    for name, data in iter_rar_files(archive, UIEB_PASSWORD):  # pass 1: score every picture without writing anything to disk
        image = small_float(Image.open(io.BytesIO(data)))
        names.append(name)
        stats.append((contrast(image), abs(colour_cast(image)), dark_channel_mean(image)))
    stats = np.array(stats)
    # worse = lower contrast, stronger colour cast, higher dark channel (more veil)
    degradation = rank_average([stats[:, 0]], [stats[:, 1], stats[:, 2]])
    order = np.argsort(degradation)
    picks = [order[i] for i in dict.fromkeys(np.linspace(0, len(order) - 1, how_many).round().astype(int))]
    levels = dict(zip(picks, tertile_levels([degradation[i] for i in picks])))
    wanted = {names[i]: i for i in picks}
    manifest, sheet = [], []
    for name, data in iter_rar_files(archive, UIEB_PASSWORD, wanted=set(wanted)):  # pass 2: decode and save only the picks
        i = wanted[name]
        out = CURATED / "water" / levels[i] / f"uieb_{Path(name).name}"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        manifest.append({"medium": "water", "source": "UIEB-raw", "id": Path(name).stem, "level": levels[i],
                         "score_degradation": f"{degradation[i]:.4f}", "contrast": f"{stats[i, 0]:.4f}", "colour_cast": f"{stats[i, 1]:.4f}",
                         "dark_channel": f"{stats[i, 2]:.4f}", "degraded_path": out.relative_to(CURATED), "reference_path": "", "origin": f"UIEB/raw-890.rar:{name}"})
        sheet.append((out, f"{levels[i][:3]} {Path(name).stem} {degradation[i]:.2f}", degradation[i]))
    write_manifest("water", manifest)
    sheet.sort(key=lambda s: s[2])
    contact_sheet("water", [(p, c) for p, c, _ in sheet], columns=10)


if __name__ == "__main__":
    curate_real_haze()
    curate_sots()
    curate_raindrops()
    curate_water()
