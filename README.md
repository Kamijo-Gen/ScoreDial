# ScoreDial

Code for **ScoreDial: Precise Control of Visual Attributes in Image Generation via Regressor Guidance from Limited Human Ratings** (WACV 2027).

Given a requested value *c* on a human-rating scale, ScoreDial generates an image whose attribute attains that value, using only 100–479 rated images per attribute:

1. **Attribute regressor *R*.** A frozen pretrained image encoder (DINOv2, DINOv3, or CLIP) and a ridge head fitted to the human ratings; the encoder representation and the ridge α are chosen by cross-validation.
2. **Target conditioning.** A LoRA on Stable Diffusion 3.5-medium is trained jointly with a target-value embedding, which is added to the timestep/text conditioning of every transformer block. The training images are pseudo-labelled by *R*.
3. **Error-adaptive regressor guidance.** During sampling, *R* scores the decoded clean estimate, and the clean latent is moved along the normalized gradient of (*R*(x) − *c*)² by a step proportional to |*R*(x) − *c*| and to a noise-dependent weight. Each guided step makes two passes.

## Installation

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128   # pick your CUDA build
pip install -r requirements.txt
```

The base model is [`stabilityai/stable-diffusion-3.5-medium`](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) at revision `b940f670`. Accept its license on Hugging Face and log in (`hf auth login`) before the first run. The soap regressor uses DINOv3 (`facebook/dinov3-vitb16-pretrain-lvd1689m`), which is also gated.

On one RTX 5090, ScoreDial takes about 30 s per 1024-px image at a peak of 12 GB of GPU memory.

## Data

Obtain each dataset from its authors and place it under `data/` (or point `SCOREDIAL_DATA` elsewhere):

| Attribute | Source | Files under `data/` |
|---|---|---|
| Tire slip resistance | Hayakawa, Noji & Kato, *J. Engineering Design* 36(1), 2025 | `tire/ratings.csv` (file name, `mean`), `tire/rated/` (the 118 rated images) |
| Interior visual complexity | SAVOIAS, Interior Design category (Saraee et al., arXiv:1810.01771) | `interior/global_ranking_interior_design.xlsx` (from `Ground truth/xlsx/`), `interior/images/` (the `Images/Interior Design` folder) |
| Food desire-to-eat | CROCUFID (Toet et al., *Front. Psychol.* 10:58, 2019) | `food/info.xlsx`, `food/images/` (an image folder such as `items_plated`; the 479 rated images are picked via `info.xlsx`) |
| Soap translucency | Translucent Image Dataset (Liao, Sawayama & Xiao, *PLoS Comput. Biol.* 19(2), 2023) | `soap/Experiment2/` (from `Psychophysics_Data/`), `soap/Experiment_Stimuli/` |

`scoredial/data.py` maps every label to [0, 1]: tire and food by their observed range, interior by its 0–100 scale, and soap by per-observer min–max normalization followed by averaging (Liao et al.'s protocol).

Interior and food train their generators on the rated images. Tire and soap train on larger pools built by `scripts/prepare_pools.py`:
- tire: the raw patent line drawings upsampled to 1024 px (we used Topaz Gigapixel), restricted to the 534 that appear in the rated archive, median-filtered (7×7) and standardized per image to mean 128 / SD 55 before clipping to 0–255, plus flips of the images whose *R* score lies in the tails, so the tire regressor must be fitted first;
- soap: a seeded random sample of 800 images from the dataset's 1024-px release.

## Pretrained weights

The regressor heads, the ScoreDial LoRAs with their value embeddings, and the baselines' LoRAs for all four attributes are available as `scoredial_weights.zip` from the [GitHub releases](https://github.com/Kamijo-Gen/ScoreDial/releases) of this repository. Unpack the archive into `outputs/`:

```
outputs/regressors/<attribute>.json
outputs/models/scoredial_<attribute>/   LoRA adapter and value_embedding.{safetensors,json}
outputs/models/baseline_<attribute>/    LoRA adapter
```

## Usage

Generate with the pretrained weights:

```bash
python scripts/generate.py tire                  # c in {0.1, 0.3, 0.5, 0.7, 0.9} x 10 seeds -> outputs/generations/scoredial/full/tire/
python scripts/generate.py tire --targets 0.25 --seeds 7 --out outputs/custom   # any requested value
```

Reproduce the experiments from scratch, per attribute (shown for tire):

```bash
python scripts/fit_regressor.py tire                                   # choose and fit R (Sec. 4.2)
python scripts/prepare_pools.py tire --originals DIR --upsampled DIR   # tire and soap only; soap: --archive FILE
python scripts/train_lora.py tire                                      # ScoreDial's LoRA and value embedding
python scripts/train_lora.py tire --no-value-embedding                 # the baselines' LoRA
python scripts/generate.py tire                                        # ScoreDial (Table 1)
for m in numeric_prompt mpgd mpgd+ ug; do python scripts/run_baselines.py tire $m; done   # baselines (Table 1)
for v in guidance_only conditioning_only raw_gradient no_error_factor; do
  python scripts/generate.py tire --variant $v                         # ablations (Table 2)
done
python scripts/evaluate.py tire                                        # R-MAE, Spearman rho, auditor displacement
```

Every setting (prompts, training steps, sampler coefficients, seeds, the baselines' strength ladder) is in `scoredial/config.py` and `scoredial/sampler.py`.

**Reproducibility.** Guided sampling back-propagates through *R*'s antialiased bicubic resize, whose CUDA backward (`upsample_bicubic2d_aa_backward`) has no deterministic implementation in PyTorch, even with `torch.use_deterministic_algorithms(True)`. Repeated runs of the same image therefore differ slightly: in our runs, *R* of a single image differed by up to 0.02, and the *R*-MAE of a 50-image grid by less than 0.001. Training is not bit-reproducible either.

## Generation manifests

`manifests/` records every generated grid reported in the paper, in the layout and format the scripts above write under `outputs/` (the images themselves are not included):

```
manifests/generations/<grid>/<attribute>/records.json       target, seed, file name and R of the saved PNG, per image
manifests/generations/<mpgd|ug>/<attribute>/selection.json  R(x) - c on the validation endpoints at every ladder strength, and the chosen strength
manifests/evaluation/<attribute>.json                       per grid: R-MAE, Spearman rho, and the auditor's A(0.9) - A(0.1) per seed
```

`<grid>` is `scoredial/<variant>` (`full` and the four ablations), `numeric_prompt`, `mpgd`, `mpgd+` or `ug`. The R-agreement numbers in Tables 1–2 follow from the records with `scoredial.evaluation.agreement`, and the auditor displacements are the means and population SDs of the per-seed values.

## Citation

```bibtex
@inproceedings{tang2027scoredial,
  title     = {ScoreDial: Precise Control of Visual Attributes in Image Generation via Regressor Guidance from Limited Human Ratings},
  author    = {Tang, Jianbo and Ishii, Keita and Sakamoto, Hiroki and Kikuchi, Manabu and Kitamura, Katsuma and
               Semba, Chiho and Nishiyama, Kenta and Mori, Teppei and Kiwaki, Yukihiro and Kato, Takeo},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## License

The code is released under the MIT License (see [LICENSE](LICENSE)). Stable Diffusion 3.5 weights, and models derived from them, are subject to the [Stability AI Community License](https://stability.ai/community-license-agreement).
