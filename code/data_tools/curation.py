"""Shared helpers for building the curated data set: tertile levels, hard links, manifests and contact sheets."""
import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"
EXISTING = DATA / "existing"
CURATED = DATA / "curated"
LEVELS = ("mild", "medium", "heavy")


def tertile_levels(scores):
    """Label each score mild / medium / heavy by which third of the sorted scores it falls in (higher score = heavier)."""
    order = np.argsort(scores)
    labels = [None] * len(scores)
    for rank, index in enumerate(order):
        labels[index] = LEVELS[min(2, rank * 3 // len(scores))]
    return labels


def link_or_copy(source, destination):
    """Put `source` at `destination` without using extra disk space (a hard link). Falls back to a copy across disks."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        destination.write_bytes(Path(source).read_bytes())


def write_manifest(name, rows):
    """Save a list of dicts as data/curated/manifest_<name>.csv so every picked file can be traced to its source."""
    CURATED.mkdir(parents=True, exist_ok=True)
    path = CURATED / f"manifest_{name}.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path.relative_to(PROJECT)} ({len(rows)} rows)")


def contact_sheet(name, items, columns=8, thumb=170):
    """Make a grid of small thumbnails. `items` is a list of (image_path_or_PIL_image, caption)."""
    sheet_dir = CURATED / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    rows = -(-len(items) // columns)
    sheet = Image.new("RGB", (columns * thumb, rows * (thumb + 16)), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for index, (picture, caption) in enumerate(items):
        image = picture if isinstance(picture, Image.Image) else Image.open(picture)
        image = image.convert("RGB")
        image.thumbnail((thumb - 4, thumb - 4))
        x, y = (index % columns) * thumb, (index // columns) * (thumb + 16)
        sheet.paste(image, (x + 2, y + 2))
        draw.text((x + 3, y + thumb), caption[:28], fill=(210, 210, 210))
    path = sheet_dir / f"{name}.jpg"
    sheet.save(path, quality=88)
    print(f"wrote {path.relative_to(PROJECT)}")
