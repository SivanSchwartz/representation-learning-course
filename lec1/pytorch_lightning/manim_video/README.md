# From PyTorch to Lightning: Manim scenes

Twelve scenes in one visual language:

| colour | means | becomes |
|---|---|---|
| blue | your model (the science) | `LightningModule` |
| green | your data | `LightningDataModule` |
| gray | engineering / plumbing | `Trainer` |
| yellow | a hook fires, or the knob being shown | callbacks, arguments |
| purple | logging | TensorBoard |

| scene | lecture folder | the picture |
|---|---|---|
| `S0_Tangled` | 1. start code | the raw script coloured by job; the lines fly into a bar chart (16 model / 6 data / 21 plumbing) |
| `S1_Model` | 2. LightningModule | real digit through 784 → 50 → 10 with real activations; `nn.Module` → `pl.LightningModule`; the loop body becomes `training_step`; the slots the Trainer calls |
| `S1b_ModelSlots` | 2, 6 | `save_hyperparameters()` → `self.hparams`, `hparams.yaml` and the checkpoint; `load_from_checkpoint` with no arguments; `example_input_array` and the model summary; weight matrices to scale (98.7% in fc1); `configure_optimizers` with a StepLR scheduler drawn as a staircase; `ReduceLROnPlateau` + `monitor` |
| `S1c_ModelAtWork` | 2, 4 | seven plain-PyTorch lines struck out; live panel of train/eval mode, gradients on/off, `self.current_epoch`, `self.global_step`; `self.log` routed to progress bar / logger / `callback_metrics` (and why `monitor=` works); `predict_step` on real test digits, one real mistake |
| `S2_Trainer` | 3. Lightning Trainer | the loop as a dial, sealed in `pl.Trainer`; `max_epochs`, `precision`; fit / validate / test |
| `S2b_TrainerDebug` | 3 | `num_sanity_val_steps`, `fast_dev_run`, `limit_train_batches`, `overfit_batches` (real curve: 2.29 → 0.009) |
| `S2c_TrainerLoop` | 3 | `val_check_interval`, `check_val_every_n_epoch`, `accumulate_grad_batches` (step lamp every 4 batches), `gradient_clip_val` norm (circle) vs `gradient_clip_algorithm="value"` (square) |
| `S2d_TrainerScale` | 10. Multi-GPU | `accelerator` options; `devices=[0, 1]` splits the data 25k/25k; DDP all-reduce; `sync_dist=True`; `DeepSpeedStrategy()` ZeRO-2 sharding; precision strings |
| `S3_DataModule` | 5. DataModule | prepare_data → setup → dataloader belts of real digits; `trainer.fit(model, dm)` and `trainer.test(model, dm)` |
| `S4_TensorBoard` | 8. Logging TensorBoard | 2,346 step losses, of which `log_every_n_steps=50` sends 46; `on_epoch=True` averages; `tb_logs/version_N`; `add_image` shows a flipped 6 that reads as 9 |
| `S5_Callbacks` | 7. Callbacks | hooks as sockets on the dial; call order at `on_train_epoch_end`; EarlyStopping on the real val_loss curve (stops at epoch 13) |
| `S6_Recap` | 6. Restructuring | `train.py` coloured by which file each line comes from |

Narration cues are the `# VO:` comments inside each scene.

## Render

```bash
brew install ffmpeg pango pkg-config          # macOS, once
pip install "manim==0.22.0" "pillow<12"        # Python 3.10+
manim -pql lightning_story.py S2b_TrainerDebug # quick preview of one scene
manim -qh  lightning_story.py -a               # all scenes, 1080p
```

To join the rendered scenes into one video, in the order above:

```bash
cd media/videos/lightning_story/1080p60
for s in S0_Tangled S1_Model S1b_ModelSlots S1c_ModelAtWork S2_Trainer S2b_TrainerDebug \
         S2c_TrainerLoop S2d_TrainerScale S3_DataModule S4_TensorBoard S5_Callbacks S6_Recap; do
  echo "file '$s.mp4'"; done > list.txt
ffmpeg -f concat -safe 0 -i list.txt -c copy lightning_story.mp4
```

`pillow<12` matters: manim 0.22 crashes on images with Pillow 12+.
No LaTeX needed. Fonts: Menlo + Palatino on macOS, DejaVu elsewhere.

## Data

`assets/lightning_story_data.npz` holds real MNIST digits plus the curves and
activations from a numpy re-run of the lecture's exact model (784-50-10, ReLU,
Adam 1e-3, batch 64, 50k/10k split, 20 epochs), and a 200-epoch run on a single
batch for `overfit_batches`. To use your own run instead, replace `step_losses`,
`epoch_train` and `epoch_val` with the values from your TensorBoard logs.
