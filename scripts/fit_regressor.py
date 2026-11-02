"""Choose and fit the attribute regressor R from the human ratings.

    python scripts/fit_regressor.py tire        # or interior / food / soap / all

Writes outputs/regressors/<attribute>.json, the ridge head that R loads, and
outputs/regressors/<attribute>_selection.json with the nested-CV estimate of the selection
procedure and the full selection grid.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scoredial import data, selection                 # noqa: E402
from scoredial.config import ATTRIBUTES, OUTPUTS      # noqa: E402
from scoredial.regressor import save_head             # noqa: E402


def fit(attribute: str, device: str):
    ratings = data.load_ratings(attribute)
    features = selection.candidate_features(ratings.paths, device=device)
    nested = selection.nested_cv(features, ratings.labels)
    chosen = selection.select(features, ratings.labels)
    winner = chosen["winner"]
    head = selection.fit_head(features, ratings.labels, winner["candidate"], winner["alpha"])
    out = OUTPUTS / "regressors"
    out.mkdir(parents=True, exist_ok=True)
    save_head(head, out / f"{attribute}.json")
    (out / f"{attribute}_selection.json").write_text(json.dumps(dict(nested_cv=nested, selection=chosen), indent=1))
    print(f"{attribute}: nested-CV R^2 {nested['r2_mean']:.3f} +- {nested['r2_std']:.3f}; "
          f"chosen {winner['candidate']} alpha={winner['alpha']:g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("attribute", choices=(*ATTRIBUTES, "all"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for attribute in (ATTRIBUTES if args.attribute == "all" else [args.attribute]):
        fit(attribute, args.device)


if __name__ == "__main__":
    main()
