# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Teaching code for a representation-learning course, built on PyTorch Lightning with MNIST → Fashion-MNIST transfer experiments. Each `lecN/` folder is a **self-contained snapshot** for one lecture. Lecture N starts as a copy of lecture N-1 and then changes. The folders share no code and import nothing from each other. When you change something, edit only the lecture you were asked about. Don't factor shared code out of the folders, and don't backport fixes to earlier lectures unless asked.

- `lec1/pytorch_lightning/` holds numbered steps (`1. start code` … `10. Multi-GPU`) that turn a plain PyTorch script into a Lightning project. The folder names contain spaces and dots, so quote them in paths. `manim_video/` holds Manim scenes for that lecture. Its README has the render commands; it needs `manim==0.22.0` and `pillow<12`.
- `lec2`–`lec4` share one layout: `config.py` (all hyperparameters and mode lists), `train.py`, `model.py`, `datamodule.py`, `callbacks.py`, and from lec3 on also `augmentations.py` and `utils.py`. lec4 adds `losses.py`.
- `colab_train.ipynb` (repo root) sparse-checks-out one lecture folder on Colab, installs its `requirements.txt`, starts TensorBoard on `tb_logs`, and runs `python train.py --mode {MODE}`. Its markdown and the `LECTURE` comment still list only lec1–lec3. Update them when a new lecture is added.

There is no test suite, linter, or build step.

## Running

Run commands from inside the lecture folder. `config.py` is imported by relative name, and the paths `dataset/`, `tb_logs/` and `mnist_encoder.pt` are relative to that folder.

```bash
cd lec4
pip install -r requirements.txt                 # pytorch-lightning, torchmetrics, tensorboard, matplotlib
python train.py                                 # runs config.MODES in order
python train.py --mode pretrain_simclr linear_probe
tensorboard --logdir tb_logs
```

`--mode` (lec3+) accepts several modes and runs them in order. Each `train.py` invocation logs to `tb_logs/<timestamp>/<mode>`. For a quick smoke test, lower `NUM_EPOCHS` / `TARGET_NUM_EPOCHS` / `N_TARGET_TRAIN` in `config.py`. `ACCELERATOR="auto"` with `PRECISION="16-mixed"` runs on the CPU on a Mac and on the GPU on Colab.

## Experiment pipeline (lec3/lec4)

The modes are a two-stage transfer pipeline, and the stages communicate through a file:

1. **Pretrain on `SOURCE_DATASET` (MNIST).** These modes are in `config.PRETRAIN_MODES`. The modes in `SAVE_ENCODER_MODES` (`pretrain` = AutoEncoder, `pretrain_supervise_encoder`, `pretrain_simclr`) save **only** `model.encoder.state_dict()` to `ENCODER_PATH`. Decoder, projector, and head weights are discarded. Each run overwrites the previous encoder file.
2. **Transfer to `TARGET_DATASET` (Fashion-MNIST).** `linear_probe` (encoder frozen, also kept in eval mode) and `finetune` (all weights, `FINETUNE_LEARNING_RATE`) build an `EncoderClassifier` that loads that file. They therefore evaluate whichever pretrain ran last.
3. `pretrain_supervised` and `end_to_end` are ResNet-18 `NN` baselines. They don't use the encoder file.

`utils.build_model(mode)` maps each mode to its model. Every encoder-based model must use the same `LATENT_DIM`, or the encoder file won't load. Adding a mode means updating `utils.MODES` and `build_model`, `config.PRETRAIN_MODES` / `SAVE_ENCODER_MODES` where the mode belongs in them, the mode branches in `train.py` (results printing, SimCLR-specific batch settings), and the mode list in the Colab notebook.

SimCLR specifics (lec4): `pretrain_simclr` uses `SIMCLR_BATCH_SIZE`. It sets `two_views=True` on `MnistDataModule`, so each item is `(xi, xj)` from `ContrastiveTransformations`, val and test use the same augmentation, and `drop_last` is on for training. `CONTRASTIVE_LOSS` selects `nt_xent` or `triplet` from `losses.py`. Both losses treat the other rows of the batch as negatives.

`MnistDataModule` uses a fixed seeded 50k/10k split, and every run calls `pl.seed_everything(42)`. Keep both so results stay comparable across modes.

## Conventions

- Code and comments are written for students: short, plain sentences that explain *why*. Match that style.
- Commit messages are a single sentence of the form "Add X so Y."
- Generated artifacts (`dataset/`, `tb_logs/`, `*.ckpt`, `*.pt`) are gitignored. Don't commit them.
