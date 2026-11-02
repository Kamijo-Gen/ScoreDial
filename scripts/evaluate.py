"""Evaluate every generated grid of one attribute: agreement with R and auditor displacement.

    python scripts/evaluate.py tire

Fits the SigLIP auditor to the human ratings and reports its nested-CV R^2. Then, for each grid
under outputs/generations/ (any directory with a records.json for the attribute), reports R-MAE
with the SD of the per-seed means, Spearman rho, and the per-seed A(0.9) - A(0.1) (mean +- SD).
Grids without both endpoints for every seed are skipped. Writes outputs/evaluation/<attribute>.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scoredial import data                                                   # noqa: E402
from scoredial.config import ATTRIBUTES, OUTPUTS, TARGETS                    # noqa: E402
from scoredial.evaluation import (Auditor, agreement, endpoint_displacement,  # noqa: E402
                                  load_siglip, siglip_features)

GENERATIONS = OUTPUTS / "generations"


def evaluate_grid(records, grid: Path, auditor, siglip, device):
    """Agreement and auditor displacement of one grid, or None if an endpoint image is missing."""
    cells = {(round(r["target"], 3), r["seed"]): grid / r["file"] for r in records}
    seeds = sorted({r["seed"] for r in records})
    low, high = TARGETS[0], TARGETS[-1]
    if not all((low, s) in cells and (high, s) in cells for s in seeds):
        return None
    result = agreement([r["target"] for r in records], [r["score"] for r in records], [r["seed"] for r in records])
    result["auditor_displacement"] = endpoint_displacement(
        auditor, [str(cells[(low, s)]) for s in seeds], [str(cells[(high, s)]) for s in seeds],
        device=device, siglip=siglip)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("attribute", choices=ATTRIBUTES)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ratings = data.load_ratings(args.attribute)
    siglip = load_siglip(args.device)
    features = siglip_features(ratings.paths, device=args.device, siglip=siglip)
    auditor = Auditor.fit(features, ratings.labels)
    nested = Auditor.nested_cv_r2(features, ratings.labels)
    report = dict(auditor=dict(nested_cv=nested, alpha=auditor.alpha), grids={})
    print(f"auditor nested-CV R^2 {nested['r2_mean']:.3f}")

    for records_file in sorted(GENERATIONS.glob(f"**/{args.attribute}/records.json")):
        grid = records_file.parent
        name = str(grid.relative_to(GENERATIONS).parent)
        result = evaluate_grid(json.loads(records_file.read_text()), grid, auditor, siglip, args.device)
        if result is None:
            print(f"{name:32s} skipped: not every seed has both endpoint images")
            continue
        report["grids"][name] = result
        d = result["auditor_displacement"]
        print(f"{name:32s} R-MAE {result['r_mae']:.4f}+-{result['r_mae_seed_sd']:.4f}  "
              f"rho {result['spearman']:.3f}  A(0.9)-A(0.1) {d['mean']:+.3f}+-{d['sd']:.3f}")

    out = OUTPUTS / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.attribute}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
