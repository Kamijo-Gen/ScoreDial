"""Generate ScoreDial images for requested values, and the ablated samplers.

    python scripts/generate.py tire                                     # 5 targets x 10 test seeds
    python scripts/generate.py tire --variant guidance_only             # no value embedding
    python scripts/generate.py tire --variant conditioning_only         # no regressor guidance
    python scripts/generate.py tire --variant raw_gradient              # step along the raw gradient
    python scripts/generate.py tire --variant no_error_factor           # step size without |R(x) - c|
    python scripts/generate.py tire --targets 0.25 --seeds 7 --out outputs/custom   # any requested value

Reads outputs/models/scoredial_<attribute>/ and outputs/regressors/<attribute>.json. Writes
t<c>_s<seed>.png and records.json (R measured on each saved PNG) to
outputs/generations/scoredial/<variant>/<attribute>/, or to --out. records.json is rewritten on
every run, so give runs outside the standard grid their own --out.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scoredial import model                                           # noqa: E402
from scoredial.config import ATTRIBUTES, OUTPUTS, TARGETS, TEST_SEEDS  # noqa: E402
from scoredial.evaluation import load_png                             # noqa: E402
from scoredial.regressor import Regressor                             # noqa: E402
from scoredial.sampler import GuidanceConfig, sample                  # noqa: E402

VARIANTS = {
    "full": GuidanceConfig(),
    "guidance_only": GuidanceConfig(),   # same sampler; the value embedding is switched off below
    "conditioning_only": GuidanceConfig(lambda_main=0.0, lambda_tail=0.0),
    "raw_gradient": GuidanceConfig(normalize_gradient=False),
    "no_error_factor": GuidanceConfig(error_factor=False),
}


def save_png(image: torch.Tensor, path: Path) -> None:
    """Write a (1, 3, H, W) image in [-1, 1] as an 8-bit PNG."""
    array = (image[0].permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5
    Image.fromarray(array.clip(0, 255).astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("attribute", choices=ATTRIBUTES)
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--weights", type=Path, default=None, help="default: outputs/models/scoredial_<attribute>")
    parser.add_argument("--targets", type=float, nargs="+", default=TARGETS)
    parser.add_argument("--seeds", type=int, nargs="+", default=TEST_SEEDS)
    parser.add_argument("--out", type=Path, default=None,
                        help="default: outputs/generations/scoredial/<variant>/<attribute>")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    weights = args.weights or OUTPUTS / "models" / f"scoredial_{args.attribute}"
    transformer, embedding, stats = model.load_scoredial(weights, device=device)
    text = model.encode_prompt(ATTRIBUTES[args.attribute]["prompt"], device=device)
    vae, scheduler = model.load_vae(device=device), model.load_scheduler()
    regressor = Regressor.load(OUTPUTS / "regressors" / f"{args.attribute}.json", device=device).eval()

    out = args.out or OUTPUTS / "generations" / "scoredial" / args.variant / args.attribute
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for target in args.targets:
        conditioning_value = None if args.variant == "guidance_only" else target
        model.set_target(embedding, conditioning_value, stats, device)
        for seed in args.seeds:
            image, _ = sample(target, seed, transformer=transformer, vae=vae, scheduler=scheduler, text=text,
                              regressor=regressor, guidance=VARIANTS[args.variant], device=device)
            path = out / f"t{target:.3f}_s{seed:03d}.png"
            save_png(image, path)
            with torch.no_grad():
                score = float(regressor(load_png(path, device)).flatten()[0])
            records.append(dict(target=target, seed=seed, file=path.name, score=score))
            print(f"{args.attribute} {args.variant} c={target:g} seed={seed}  R={score:.4f}", flush=True)
            (out / "records.json").write_text(json.dumps(records, indent=1))


if __name__ == "__main__":
    main()
