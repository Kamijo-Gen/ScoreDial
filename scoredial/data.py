"""Rated images and their labels, and the image pools the generators are trained on.

Every label is mapped to [0, 1], the scale on which requested values are given. Expected layout
under ``data/`` (README lists the sources):

    tire/ratings.csv              file name (first column) and mean five-point rating (``mean``)
    tire/rated/                   the 118 rated tread images
    tire/train/                   training pool, built by scripts/prepare_pools.py
    interior/global_ranking_interior_design.xlsx
                                  SAVOIAS ground truth (column ``gt``; row i is image i.jpg)
    interior/images/              SAVOIAS "Images/Interior Design" (0.jpg ... 99.jpg)
    food/info.xlsx                CROCUFID metadata
    food/images/                  CROCUFID images named <image nr>.<ext>, e.g. "items_plated"
    soap/Experiment2/Exp2_*.csv   Translucent Image Dataset, Experiment 2 responses (20 observers)
    soap/Experiment_Stimuli/      Translucent Image Dataset, Experiment 2 stimuli
    soap/train/                   training pool, built by scripts/prepare_pools.py
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DATA

RATED_IMAGES = {"tire": 118, "interior": 100, "food": 479, "soap": 150}
TRAINING_IMAGES = {"tire": 984, "interior": 100, "food": 479, "soap": 800}
# Tire and soap train on pools built by prepare_pools.py; interior and food on their rated images.
BUILT_POOLS = {"tire": "tire/train", "soap": "soap/train"}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


@dataclass
class Ratings:
    attribute: str
    paths: list[str]
    labels: np.ndarray   # in [0, 1]


def _rescale(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return (values - low) / (high - low)


def _tire(root: Path) -> Ratings:
    """Mean Likert ratings, mapped to [0, 1] by their observed range."""
    import pandas as pd

    table = pd.read_csv(root / "tire" / "ratings.csv")
    names = table[table.columns[0]]
    means = table["mean"].to_numpy(float)
    paths = [str(root / "tire" / "rated" / name) for name in names]
    return Ratings("tire", paths, _rescale(means, means.min(), means.max()))


def _interior(root: Path) -> Ratings:
    """SAVOIAS complexity scores, mapped to [0, 1] by the endpoints of their 0-100 scale."""
    import pandas as pd

    scores = pd.read_excel(root / "interior" / "global_ranking_interior_design.xlsx")["gt"].to_numpy(float)
    paths = [str(root / "interior" / "images" / f"{i}.jpg") for i in range(len(scores))]
    return Ratings("interior", paths, _rescale(scores, 0.0, 100.0))


def _food(root: Path) -> Ratings:
    """CROCUFID desire-to-eat ratings of the images present, mapped to [0, 1] by their observed range."""
    import openpyxl

    rows = openpyxl.load_workbook(root / "food" / "info.xlsx", read_only=True)["Sheet1"].iter_rows(values_only=True)
    header = next(rows)
    id_column, rating_column = header.index("image nr"), header.index("Average_Desire_ALL")
    images = {int(p.stem): p for p in (root / "food" / "images").iterdir() if p.is_file() and p.stem.isdigit()}
    rated = sorted(
        (int(row[id_column]), float(row[rating_column])) for row in rows
        if isinstance(row[id_column], (int, float)) and isinstance(row[rating_column], (int, float))
        and int(row[id_column]) in images)
    ratings = np.asarray([rating for _, rating in rated], dtype=float)
    paths = [str(images[image_id]) for image_id, _ in rated]
    return Ratings("food", paths, _rescale(ratings, ratings.min(), ratings.max()))


def _soap(root: Path) -> Ratings:
    """Translucency of the 150 real photographs, following Liao et al.: each observer's 1-7
    responses are min-max normalized, then averaged over the 20 observers."""
    import pandas as pd

    response_files = sorted((root / "soap" / "Experiment2").glob("Exp2_*.csv"))
    if len(response_files) != 20:
        raise RuntimeError(f"expected 20 Experiment-2 response files, found {len(response_files)}")
    normalized = defaultdict(list)
    for file in response_files:
        table = pd.read_csv(file)
        real = table[table["ground_truth"] == "1"]
        responses = real["translucency_score.response"].to_numpy(float)
        scaled = (responses - responses.min()) / (responses.max() - responses.min())
        for name, value in zip(real["image_name"], scaled):
            normalized[str(name)].append(float(value))
    stimuli = {p.name: p for pattern in ("**/*.jpg", "**/*.png")
               for p in (root / "soap" / "Experiment_Stimuli").glob(pattern)}
    rated = sorted((name, float(np.mean(values))) for name, values in normalized.items()
                   if len(values) == 20 and name.startswith("Group1_") and name in stimuli)
    return Ratings("soap", [str(stimuli[name]) for name, _ in rated],
                   np.asarray([value for _, value in rated], dtype=float))


_LOADERS = {"tire": _tire, "interior": _interior, "food": _food, "soap": _soap}


def load_ratings(attribute: str, root: Path = DATA) -> Ratings:
    """Rated images and labels of one attribute; checks the image count and that every file exists."""
    ratings = _LOADERS[attribute](Path(root))
    if len(ratings.paths) != RATED_IMAGES[attribute]:
        raise RuntimeError(f"{attribute}: expected {RATED_IMAGES[attribute]} rated images, "
                           f"found {len(ratings.paths)}")
    missing = [p for p in ratings.paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{attribute}: {len(missing)} rated images missing, e.g. {missing[0]}")
    return ratings


def training_images(attribute: str, root: Path = DATA) -> list[str]:
    """Sorted paths of the images the attribute's generator is trained on."""
    if attribute in BUILT_POOLS:
        folder = Path(root) / BUILT_POOLS[attribute]
        paths = sorted(str(p) for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    else:
        paths = sorted(load_ratings(attribute, root).paths)
    if len(paths) != TRAINING_IMAGES[attribute]:
        raise RuntimeError(f"{attribute}: expected {TRAINING_IMAGES[attribute]} training images, found {len(paths)}")
    return paths
