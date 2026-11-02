"""Paths, model versions, and experiment settings shared by all scripts."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("SCOREDIAL_DATA", REPO / "data"))
OUTPUTS = Path(os.environ.get("SCOREDIAL_OUTPUTS", REPO / "outputs"))

BASE_MODEL = "stabilityai/stable-diffusion-3.5-medium"
BASE_REVISION = "b940f670f0eda2d07fbb75229e779da1ad11eb80"

# Requested values and test seeds; ScoreDial, the baselines and the ablations all use this grid.
TARGETS = (0.1, 0.3, 0.5, 0.7, 0.9)
TEST_SEEDS = tuple(range(10))
# Seeds used only to choose the baselines' guidance strength.
VALIDATION_SEEDS = tuple(range(1001, 1011))

# Guidance strengths generated for MPGD-flow and UG-flow, and the ones each may select.
STRENGTH_LADDER = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
ADMISSIBLE_STRENGTHS = {"mpgd": STRENGTH_LADDER, "ug": (1.0, 3.0, 10.0, 30.0)}

# LoRA training, identical for every attribute except the number of steps.
TRAINING = dict(rank=32, batch_size=4, learning_rate=1e-4, size=1024, seed=0)

ATTRIBUTES = {
    "tire": dict(
        prompt="a tire tread pattern, top-down view, grayscale",
        numeric_prompt="a tire tread pattern, top-down view, grayscale, with perceived slip resistance",
        train_steps=3000,
    ),
    "interior": dict(
        prompt="a photograph of a room interior",
        numeric_prompt="a photograph of a room interior, with perceived visual complexity",
        train_steps=3000,
    ),
    "food": dict(
        prompt="a plate of food on white background",
        numeric_prompt="a plate of food on white background, with perceived desire to eat",
        train_steps=2000,
    ),
    "soap": dict(
        prompt="a bar of soap",
        numeric_prompt="a bar of soap, with perceived translucency",
        train_steps=6000,
    ),
}


def numeric_prompt(attribute: str, target: float) -> str:
    """Prompt of the Numeric-prompt baseline: the requested value written into the text."""
    return f"{ATTRIBUTES[attribute]['numeric_prompt']} rated {target:.1f} on a zero-to-one scale"
