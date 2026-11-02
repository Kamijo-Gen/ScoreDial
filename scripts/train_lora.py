"""Train an attribute's LoRA with the target-value embedding (ScoreDial) or without it (baselines).

    python scripts/train_lora.py tire                        # -> outputs/models/scoredial_tire
    python scripts/train_lora.py tire --no-value-embedding   # -> outputs/models/baseline_tire

Settings come from scoredial/config.py: 3000/3000/2000/6000 steps for tire/interior/food/soap,
rank 32, batch 4, AdamW at a constant learning rate of 1e-4, 1024-px images. ScoreDial's training
needs the fitted regressor (outputs/regressors/<attribute>.json) for pseudo-labels.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scoredial import data                               # noqa: E402
from scoredial.config import ATTRIBUTES, OUTPUTS         # noqa: E402
from scoredial.regressor import Regressor                # noqa: E402
from scoredial.train import train                        # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("attribute", choices=ATTRIBUTES)
    parser.add_argument("--no-value-embedding", action="store_true")
    parser.add_argument("--steps", type=int, default=None, help="default: the attribute's setting in config.py")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    settings = ATTRIBUTES[args.attribute]
    with_embedding = not args.no_value_embedding
    out = args.out or OUTPUTS / "models" / f"{'scoredial' if with_embedding else 'baseline'}_{args.attribute}"
    # Only ScoreDial's value embedding uses pseudo-labels from R.
    regressor = (Regressor.load(OUTPUTS / "regressors" / f"{args.attribute}.json", device=args.device).eval()
                 if with_embedding else None)
    train(data.training_images(args.attribute), settings["prompt"], out,
          steps=args.steps or settings["train_steps"], regressor=regressor,
          value_embedding=with_embedding, device=args.device)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
