"""Run a baseline on the test grid, on the baseline's own domain LoRA.

    python scripts/run_baselines.py tire numeric_prompt
    python scripts/run_baselines.py tire mpgd      # choose the strength, then generate the test grid
    python scripts/run_baselines.py tire mpgd+     # MPGD-flow one ladder rung above the chosen strength
    python scripts/run_baselines.py tire ug

The guidance strength of MPGD-flow and UG-flow is chosen on the validation seeds at the endpoints
c = 0.1 and 0.9: the admissible strength with the lowest mean |R(x) - c| (config.ADMISSIBLE_STRENGTHS).
The choice is saved in outputs/generations/<mpgd|ug>/<attribute>/selection.json and reused.

Reads outputs/models/baseline_<attribute>/ and outputs/regressors/<attribute>.json. Writes images
and records.json (R measured on each saved PNG) to outputs/generations/<method>/<attribute>/.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scoredial import model                                                       # noqa: E402
from scoredial.baselines import guided_sample, select_strength, stronger_rung     # noqa: E402
from scoredial.config import (ATTRIBUTES, OUTPUTS, STRENGTH_LADDER, TARGETS,      # noqa: E402
                              TEST_SEEDS, VALIDATION_SEEDS, numeric_prompt)
from scoredial.evaluation import load_png                                         # noqa: E402
from scoredial.regressor import Regressor                                         # noqa: E402
from scoredial.sampler import GuidanceConfig, sample                              # noqa: E402

NO_GUIDANCE = GuidanceConfig(lambda_main=0.0, lambda_tail=0.0)
GENERATIONS = OUTPUTS / "generations"


def save_and_score(image, path, regressor, device) -> float:
    """Save a (1, 3, H, W) image in [-1, 1] as PNG and return R of the saved file."""
    save_image(((image[0] + 1.0) / 2.0).clamp(0, 1), path)
    with torch.no_grad():
        return float(regressor(load_png(path, device))[0].cpu())


def run_numeric_prompt(attribute, out, transformer, vae, scheduler, regressor, device):
    records = []
    for target in TARGETS:
        text = model.encode_prompt(numeric_prompt(attribute, target), device=device)
        for seed in TEST_SEEDS:
            image, _ = sample(target, seed, transformer=transformer, vae=vae, scheduler=scheduler,
                              text=text, regressor=regressor, guidance=NO_GUIDANCE, device=device)
            path = out / f"t{target:.3f}_s{seed:03d}.png"
            score = save_and_score(image, path, regressor, device)
            records.append(dict(target=target, seed=seed, file=path.name, score=score))
            print(f"numeric_prompt c={target:g} seed={seed}  R={score:.4f}", flush=True)
    return records


def run_guided(method, strength, targets, seeds, out, components, device):
    transformer, vae, scheduler, text, regressor = components
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for target in targets:
        for seed in seeds:
            image, _ = guided_sample(method, target, seed, strength, transformer=transformer, vae=vae,
                                     scheduler=scheduler, text=text, regressor=regressor, device=device)
            path = out / f"w{strength:g}_t{target:.3f}_s{seed:03d}.png"
            score = save_and_score(image, path, regressor, device)
            records.append(dict(target=target, seed=seed, strength=strength, file=path.name, score=score))
            print(f"{method} w={strength:g} c={target:g} seed={seed}  R={score:.4f}", flush=True)
    return records


def chosen_strength(method, attribute, components, device) -> float:
    selection_file = GENERATIONS / method / attribute / "selection.json"
    if selection_file.is_file():
        return json.loads(selection_file.read_text())["selected"]
    # Every rung is generated, including those not admissible for selection, so that their
    # outputs can be inspected.
    errors = {}
    for strength in STRENGTH_LADDER:
        cells = run_guided(method, strength, (TARGETS[0], TARGETS[-1]), VALIDATION_SEEDS,
                           GENERATIONS / method / attribute / "validation", components, device)
        errors[strength] = [c["score"] - c["target"] for c in cells]
    selected = select_strength(method, errors)
    selection_file.write_text(json.dumps(
        dict(selected=selected, endpoint_errors={str(k): v for k, v in errors.items()}), indent=1))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("attribute", choices=ATTRIBUTES)
    parser.add_argument("method", choices=("numeric_prompt", "mpgd", "mpgd+", "ug"))
    parser.add_argument("--weights", type=Path, default=None, help="default: outputs/models/baseline_<attribute>")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device, attribute = args.device, args.attribute
    weights = args.weights or OUTPUTS / "models" / f"baseline_{attribute}"
    transformer = model.load_baseline(weights, device=device)
    vae, scheduler = model.load_vae(device=device), model.load_scheduler()
    regressor = Regressor.load(OUTPUTS / "regressors" / f"{attribute}.json", device=device).eval()
    out = GENERATIONS / args.method / attribute
    out.mkdir(parents=True, exist_ok=True)

    if args.method == "numeric_prompt":
        records = run_numeric_prompt(attribute, out, transformer, vae, scheduler, regressor, device)
    else:
        method = "mpgd" if args.method == "mpgd+" else args.method
        text = model.encode_prompt(ATTRIBUTES[attribute]["prompt"], device=device)
        components = (transformer, vae, scheduler, text, regressor)
        strength = chosen_strength(method, attribute, components, device)
        if args.method == "mpgd+":
            strength = stronger_rung(method, strength)
        records = run_guided(method, strength, TARGETS, TEST_SEEDS, out, components, device)
    (out / "records.json").write_text(json.dumps(records, indent=1))


if __name__ == "__main__":
    main()
