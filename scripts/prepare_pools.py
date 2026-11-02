"""Build the generator-training pools of tire and soap. Interior and food train on their rated
images, so there is nothing to build for them.

tire: --upsampled holds the raw patent line drawings upsampled to 1024 px (we used Topaz
      Gigapixel); --originals holds the 535-image archive of Hayakawa et al., whose file names
      (without extension) select the 534 tread images. Each upsampled image is median-filtered
      (7x7) and standardized per image to mean 128 and SD 55 (clipped to 0-255). Horizontal,
      vertical, and both-axis flips are added for every image whose R score on the archive image
      is <= 0.252 or >= 0.746: 984 images. R must already be fitted.

          python scripts/prepare_pools.py tire --originals DIR --upsampled DIR

soap: draws 800 photographs uniformly at random (Python random.Random(0)) from the 8,085 images of
      the Translucent Image Dataset's 1024-px release.

          python scripts/prepare_pools.py soap --archive TID_res1024_8k.zip

The pools are written to data/tire/train/ and data/soap/train/.
"""
import argparse
import random
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scoredial.config import DATA, OUTPUTS   # noqa: E402

TAIL_LOW, TAIL_HIGH = 0.252, 0.746
TIRE_MEAN, TIRE_SD = 128.0, 55.0
SOAP_POOL_SIZE = 800


def _by_stem(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".png")}


def standardize_tread(gray):
    """7x7 median filter, then per-image standardization to mean 128 / SD 55, clipped to 0-255."""
    import numpy as np
    from scipy.ndimage import median_filter

    x = median_filter(gray, size=7).astype(np.float64)
    return np.clip((x - x.mean()) / x.std() * TIRE_SD + TIRE_MEAN, 0, 255).astype(np.uint8)


def build_tire(originals: Path, upsampled: Path, out: Path, regressor_path: Path, device: str):
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    from scoredial.evaluation import load_png
    from scoredial.regressor import Regressor

    archive, drawings = _by_stem(originals), _by_stem(upsampled)
    stems = sorted(set(archive) & set(drawings))
    regressor = Regressor.load(regressor_path, device=device).eval()
    with torch.no_grad():
        scores = {s: float(regressor(load_png(archive[s], device)).flatten()[0]) for s in stems}
    tails = {s for s in stems if scores[s] <= TAIL_LOW or scores[s] >= TAIL_HIGH}
    out.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        gray = np.asarray(Image.open(drawings[stem]).convert("L"))
        Image.fromarray(standardize_tread(gray)).save(out / f"{stem}.jpg", quality=95)
        if stem in tails:
            image = Image.open(out / f"{stem}.jpg")
            ImageOps.mirror(image).save(out / f"{stem}_hf.jpg", quality=97)
            ImageOps.flip(image).save(out / f"{stem}_vf.jpg", quality=97)
            ImageOps.flip(ImageOps.mirror(image)).save(out / f"{stem}_hv.jpg", quality=97)
    low = sum(scores[s] <= TAIL_LOW for s in stems)
    print(f"tire: {len(stems)} images, {low} at <= {TAIL_LOW}, {len(tails) - low} at >= {TAIL_HIGH}, "
          f"{len(stems) + 3 * len(tails)} in {out}")


def build_soap(archive: Path, out: Path):
    with zipfile.ZipFile(archive) as handle:
        candidates = [n for n in handle.namelist() if n.endswith(".jpg") and not n.startswith("__MACOSX")]
        chosen = random.Random(0).sample(candidates, SOAP_POOL_SIZE)
        out.mkdir(parents=True, exist_ok=True)
        for name in chosen:
            (out / Path(name).name).write_bytes(handle.read(name))
    print(f"soap: {len(chosen)} of {len(candidates)} images -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("attribute", choices=("tire", "soap"))
    parser.add_argument("--originals", type=Path, help="tire: the 535-image archive of Hayakawa et al.")
    parser.add_argument("--upsampled", type=Path, help="tire: the raw line drawings upsampled to 1024 px")
    parser.add_argument("--archive", type=Path, help="soap: TID_res1024_8k.zip")
    parser.add_argument("--regressor", type=Path, default=OUTPUTS / "regressors" / "tire.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.attribute == "tire":
        if args.originals is None or args.upsampled is None:
            parser.error("tire needs --originals and --upsampled")
        build_tire(args.originals, args.upsampled, DATA / "tire" / "train", args.regressor, args.device)
    else:
        if args.archive is None:
            parser.error("soap needs --archive")
        build_soap(args.archive, DATA / "soap" / "train")


if __name__ == "__main__":
    main()
