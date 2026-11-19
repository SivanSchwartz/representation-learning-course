# AGENTS.md

This file provides guidance to AI coding agents (Codex, Claude Code) when working with code in this repository.

## What this repo is

Teaching code for a representation-learning course, built on PyTorch Lightning with MNIST and Fashion-MNIST. Each `lecN/` folder is a **self-contained snapshot** for one lecture. Lecture N starts as a copy of lecture N-1 and then changes. The folders share no code and import nothing from each other. When you change something, edit only the lecture you were asked about. Don't factor shared code out of the folders, and don't backport fixes to earlier lectures unless asked.

There is no test suite, linter, or build step.

## The lectures

| folder | topic | what it adds |
|---|---|---|
| `lec1/pytorch_lightning/` | From plain PyTorch to Lightning | Numbered steps (`1. start code` … `10. Multi-GPU`), each a runnable snapshot. The folder names contain spaces and dots, so quote them in paths. `manim_video/` holds the lecture's Manim scenes. Its README has the render commands; it needs `manim==0.22.0` and `pillow<12`. lec1 has no `requirements.txt`. |
| `lec2` | Transfer learning | ResNet-18 `NN` only. Supervised pretraining on MNIST, then `linear_probe` / `finetune` / `end_to_end` on Fashion-MNIST. You choose the runs by editing `config.MODES`, because there is no `--mode` flag. |
| `lec3` | Autoencoder pretraining and augmentation | Adds `AutoEncoder`, `EncoderClassifier`, `SupervisedEncoder`, `augmentations.py`, `utils.py` (`parse_args`, `build_model`), `--mode`, and `notebooks/03_data_augmentation_v2.ipynb`. |
| `lec4` | SimCLR and contrastive learning | Adds `SimCLR`, `losses.py` (`nt_xent_loss`, `triplet_loss`), and the two-view data path. |
| `lec5` | MoCo | Adds `MoCo` and `notebooks/05_moco_experiments.ipynb`, a student exercise notebook that imports the lecture code. It also changes the CLI: see below. |

`lec2`–`lec5` share one layout: `config.py` (all hyperparameters and mode lists), `train.py`, `model.py`, `datamodule.py` (`MnistDataModule`), and `callbacks.py`.

`colab_train.ipynb` (repo root) sparse-checks-out one lecture folder on Colab, installs its `requirements.txt`, starts TensorBoard on `tb_logs`, and runs `python train.py --mode {MODE}`. Its text and the `LECTURE` comment still list only lec1–lec3, and its mode list lacks `pretrain_simclr` and `pretrain_moco`. Update them when a lecture is added. Lecture notebooks under `lecN/notebooks/` have their own Open-in-Colab badge and setup cell.

## Running

Run commands from inside the lecture folder. `config.py` is imported by relative name, and `dataset/`, `tb_logs/` and the encoder `.pt` file are relative to that folder.

```bash
cd lec4
pip install -r requirements.txt                    # pytorch-lightning, torchmetrics, tensorboard, matplotlib
python train.py                                    # lec2–lec4: runs config.MODES in order
python train.py --mode pretrain_simclr linear_probe   # lec3–lec4: several modes, run in order
tensorboard --logdir tb_logs
```

```bash
cd lec5
python train.py                                    # one run: config.MODEL_NAME on MNIST
python train.py --mode pretrain_moco --dataset fashionmnist
python train.py --mode linear_probe --dataset fashionmnist
```

Each run logs to `tb_logs/<timestamp>/<mode>`. For a quick smoke test, lower `NUM_EPOCHS` / `TARGET_NUM_EPOCHS` / `N_TARGET_TRAIN` in `config.py`. `ACCELERATOR="auto"` with `PRECISION="16-mixed"` runs on the CPU on a Mac and on the GPU on Colab.

## Experiment pipeline

Training is two stages that pass an encoder through a file:

1. **Pretrain.** Modes listed in `config.PRETRAIN_MODES`. The ones in `SAVE_ENCODER_MODES` (`pretrain` = AutoEncoder, `pretrain_supervise_encoder`, `pretrain_simclr`, `pretrain_moco`) save **only** `model.encoder.state_dict()` to `ENCODER_PATH`. Decoder, projector, key-encoder, and head weights are discarded. Each run overwrites the previous encoder file.
2. **Evaluate the encoder.** `linear_probe` (encoder frozen and kept in eval mode) and `finetune` (all weights, `FINETUNE_LEARNING_RATE`) build an `EncoderClassifier` that loads that file. They therefore evaluate whichever pretrain ran last.
3. `pretrain_supervised` and `end_to_end` are ResNet-18 `NN` baselines. They don't use the encoder file.

Datasets differ by lecture:
- **lec2–lec4:** pretrain modes use `SOURCE_DATASET` (MNIST), and the others use `TARGET_DATASET` (Fashion-MNIST). `ENCODER_PATH` is fixed (`mnist_encoder.pt`; in lec2 `mnist_resnet18_encoder.pt`).
- **lec5:** there is no source/target split. `--dataset {mnist,fashionmnist}` sets `config.DATASET` for every mode, and `parse_args` rewrites `ENCODER_PATH` to `<dataset>_encoder.pt`. So a probe loads the encoder pretrained on the same `--dataset`. `--mode` takes one name, and its default is `config.MODEL_NAME`.

`utils.build_model(mode)` maps each mode to its model. Every encoder-based model must use the same `LATENT_DIM`, or the encoder file won't load. Adding a mode means updating `utils.MODES` and `build_model`, `config.PRETRAIN_MODES` / `SAVE_ENCODER_MODES` where the mode belongs in them, the per-mode branches in `train.py` (batch size, `two_views`, the returned metric, results printing), and the mode list in `colab_train.ipynb`.

Contrastive modes:
- `pretrain_simclr` (lec4+) uses `SIMCLR_BATCH_SIZE`, because the other rows of the batch are its negatives. `CONTRASTIVE_LOSS` selects `nt_xent` or `triplet` from `losses.py`.
- `pretrain_moco` (lec5) keeps `BATCH_SIZE` and takes negatives from a queue of `MOCO_K` keys, which a momentum key encoder updates with `MOCO_M`. `MOCO_K` must be a multiple of the batch size. Only the training step updates the key encoder and the queue. Val and test only read the queue.
- Both set `two_views=True` on `MnistDataModule`. Each item is then `(xi, xj)` from `ContrastiveTransformations`, val and test use the same augmentation (so their loss is the contrastive loss), and `drop_last` is on for training.

`MnistDataModule` uses a fixed seeded 50k/10k split, and every run calls `pl.seed_everything(42)`. Keep both so results stay comparable across modes.

## Conventions

- Code and comments are written for students: short, plain sentences that explain *why*. Match that style. The exercise notebooks ask students questions about the code, such as in `lec5/notebooks/05_moco_experiments.ipynb`. If you change that code, check that the notebook's questions still match.
- Commit messages are one short sentence, usually of the form "Add X so Y."
- Generated artifacts (`dataset/`, `tb_logs/`, `*.ckpt`, `*.pt`) are gitignored. Don't commit them.
