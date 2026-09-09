# TSMD: Temporal-Stream Modality Dropout for Robust Video Highlight Detection

Official PyTorch implementation of **TSMD: Temporal-Stream Modality Dropout for
Robust Video Highlight Detection**.

[Interactive demo](https://boyuan-ch.github.io/TSMD/) | [Code](https://github.com/boyuan-ch/TSMD)

TSMD studies video highlight detection when visual, audio, or text features are
missing at isolated timestamps or for an entire video. It augments a
TripleSumm-based detector with:

- **MPR**, a metric-aligned objective combining pointwise MSE, per-video Pearson
  correlation, and peak-oriented RankNet loss.
- **TSMD-Temporal**, independent temporal dropout in each modality stream.
- **TSMD-Stream**, uniform sampling over clean, missing-visual, missing-audio,
  and missing-text inputs.
- **TSMD-Mix**, mutually exclusive clean, temporal, and whole-stream corruption.

All variants preserve sequence length, timestamps, targets, padding masks, and
the inference architecture. Missing feature rows are set to zero during
training or robustness evaluation.

## News

- **September 2026:** Initial research code release.

## Installation

The release targets Python 3.10 and PyTorch 2.5 or newer. A CUDA 12.1
environment is provided:

```bash
git clone https://github.com/boyuan-ch/TSMD.git
cd TSMD
conda env create -f environment.yml
conda activate tsmd
```

The publication smoke tests are CPU-compatible:

```bash
python -m unittest discover -s tests -v
```

The original TripleSumm implementation was tested with PyTorch 2.5.1 and CUDA
12.1. This release was additionally validated in a clean `tsmd` environment;
see `environment.yml` and `requirements.txt` for the reproducible dependency
specification.

## Data

TSMD uses the pre-extracted multimodal features and annotations released with
TripleSumm:

- [MoSu on Hugging Face](https://huggingface.co/datasets/hminjeong/TripleSumm-MoSu)
- [Mr. HiSum on Hugging Face](https://huggingface.co/datasets/hminjeong/TripleSumm-Mr.HiSum)

Place downloaded files under `data/mosu/` and `data/mrhisum/`. Exact filenames
and layouts are listed in [data/README.md](data/README.md). The feature archives
are large and are intentionally excluded from Git.

## Training

The public runner exposes the four policies used in the paper:

```bash
# Usage: bash scripts/train.sh <dataset> <policy> [seed]
bash scripts/train.sh mosu clean 42
bash scripts/train.sh mosu temporal 42
bash scripts/train.sh mosu stream 42
bash scripts/train.sh mosu mix 42
```

Use `mrhisum` in place of `mosu` for the second dataset. The paper reports three
independent training seeds:

```bash
for seed in 42 123 2026; do
  bash scripts/train.sh mosu mix "$seed"
done
```

Set `DATA_DIR` to use a data root outside the repository:

```bash
DATA_DIR=/path/to/data bash scripts/train.sh mosu temporal 42
```

Checkpoints and logs are written to
`outputs/<dataset>/triplesumm-ranknet/mpr-<policy>-s<seed>/`. The primary
checkpoint, `best_model_ckpt.pth`, is selected by validation Kendall's
$\tau$ plus Spearman's $\rho$.

### Paper policy definitions

| Policy | Training distribution |
|---|---|
| `clean` | No simulated missingness |
| `temporal` | Uniform over 0%, 10%, 30%, and 50% independent temporal dropout |
| `stream` | Uniform over clean, missing visual, missing text, and missing audio |
| `mix` | 25% clean, 37.5% temporal, and 37.5% one missing stream |

For the temporal component of `mix`, the dropout ratio is sampled uniformly
from 10%, 30%, and 50%.

## Evaluation

Evaluate a checkpoint on clean test inputs:

```bash
bash scripts/evaluate.sh mosu \
  outputs/mosu/triplesumm-ranknet/mpr-mix-s42/best_model_ckpt.pth
```

Evaluate the paper's independent 50% frame-loss condition with a deterministic
mask seed:

```bash
python evaluate_robustness.py \
  --dataset mosu \
  --checkpoint /path/to/best_model_ckpt.pth \
  --pattern independent \
  --ratio 0.5 \
  --mask-seed 42
```

Supported evaluation patterns are `clean`, `independent`, `synchronized`,
`missing-visual`, `missing-audio`, and `missing-text`. The convenience script
runs three independent mask seeds and all three stream outages:

```bash
bash scripts/evaluate_robustness.sh mosu /path/to/best_model_ckpt.pth
```

Both evaluation entry points report Kendall's $\tau$, Spearman's $\rho$,
mAP@50, and mAP@15. Robustness masks are deterministic functions of the mask
seed and video identifier.

## Main results

Three-seed test means from the paper are shown below. Complete results and
standard deviations are reported in the paper.

| Dataset | Condition | MPR clean | TSMD policy | mAP@15 |
|---|---|---:|---|---:|
| MoSu | Clean input | 45.26 | TSMD-Mix | **45.74** |
| Mr. HiSum | Clean input | **42.31** | TSMD-Mix | 42.19 |
| MoSu | 50% independent frame loss | 40.50 | TSMD-Mix | **45.10** |
| Mr. HiSum | 50% independent frame loss | 39.85 | TSMD-Temporal | **41.60** |
| MoSu | Mean over one missing stream | 41.46 | TSMD-Stream | **43.10** |
| Mr. HiSum | Mean over one missing stream | 39.15 | TSMD-Stream | **40.40** |

## Repository layout

```text
TSMD/
├── configs/                 # MPR configurations for MoSu and Mr. HiSum
├── experiments/             # Training instrumentation used by the solver
├── models/                  # TripleSumm backbone and prediction heads
├── scripts/                 # Unified train and evaluation runners
├── tests/                   # CPU-compatible publication smoke tests
├── utils/                   # MPR losses, TSMD policies, metrics, and logging
├── dataset.py
├── evaluate_robustness.py
├── main.py
└── solver.py
```

## Relationship to TripleSumm

This repository is derived from the official
[TripleSumm](https://github.com/smkim37/TripleSumm) implementation by Sumin Kim,
Hyemin Jeong, Mingu Kang, Yejin Kim, Yoori Oh, and Joonseok Lee. TripleSumm
provides the multimodal backbone, released features, annotations, and the base
training pipeline. TSMD adds the MPR objective, temporal and stream-level
missingness policies, mixed-policy training, robustness evaluation, and the
experiments described in our paper.

The upstream MIT license and copyright notice are retained. See
[NOTICE.md](NOTICE.md) for attribution. If this repository or its derived model
is useful in your work, please cite both TSMD and TripleSumm.

## Citation

The manuscript is currently in preparation and has not yet been archived. In
the meantime, please cite this software release as follows. This entry will be
replaced with the paper citation when a public preprint is available.

```bibtex
@misc{cheng2026tsmd,
  title        = {TSMD: Temporal-Stream Modality Dropout for Robust Video Highlight Detection},
  author       = {Cheng, Bo-Yuan and Chen, Kuan-Yu and Huang, Po-Han and Li, Jeng-Lin and Tsao, Yu and Ding, Jian-Jiun},
  year         = {2026},
  note         = {Manuscript in preparation. Software available at \url{https://github.com/boyuan-ch/TSMD}}
}

@inproceedings{kim2026triplesumm,
  title     = {TripleSumm: Adaptive Triple-Modality Fusion for Video Summarization},
  author    = {Kim, Sumin and Jeong, Hyemin and Kang, Mingu and Kim, Yejin and Oh, Yoori and Lee, Joonseok},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026}
}
```

## License

This repository is released under the MIT License. See [LICENSE](LICENSE).
