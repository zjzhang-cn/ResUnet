# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Unofficial PyTorch implementation of Deep Residual U-Net and ResUNet++ for semantic segmentation. Target application is road/building extraction from aerial imagery. The codebase is experimental and hard-coded for the Massachusetts Roads Dataset (1500×1500 source images, 224×224 crops).

## Commands

```bash
# Preprocess (crops 1500×1500 images into 224×224 tiles)
python preprocess.py --config "configs/default.yaml" --train <train_dir> --valid <valid_dir>

# Train
python train.py --name "experiment_name" --config "configs/default.yaml" --epochs 75

# Resume training
python train.py --name "experiment_name" --config "configs/default.yaml" --resume <checkpoint.pt>

# Inference (single image or directory)
python inference.py -c configs/default.yaml --checkpoint <model.pt> --input <path> --output <path>

# Inference with ResUNet++ override and probability map output
python inference.py -c configs/default.yaml --checkpoint <model.pt> --input <path> --output <path> --resnet-plus-plus --save-prob

# TensorBoard
tensorboard --logdir logs/

# Run tests
pytest

# Run a single test
pytest tests/test_res_unet.py::test_resunet

# Lint (per CI)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

## Architecture

### Models (`core/`)

Three architectures with increasing complexity, all for single-channel binary segmentation output (sigmoid):

- **UNet** (`core/unet.py`) — Standard U-Net with `EncodingBlock`/`DecodingBlock` building blocks. Uses reflection padding and PReLU. **Known bug**: `UNet.forward` references `input` (the Python built-in) instead of the parameter `x`; `UNetSmall.forward` does this correctly.
- **ResUNet** (`core/res_unet.py`) — U-Net variant replacing standard conv blocks with residual blocks (`ResidualConv`). Hand-coded 4-level encoder/decoder with transposed-conv upsampling. Selected when `RESNET_PLUS_PLUS: False` in config.
- **ResUNet++** (`core/res_unet_plus.py`) — Adds Squeeze-and-Excitation channel attention, ASPP multi-scale pooling at bottleneck and output, and attention-gated skip connections. Selected when `RESNET_PLUS_PLUS: True` (default).

Shared building blocks in `core/modules.py`: `ResidualConv`, `Upsample` (transposed conv), `Upsample_` (bilinear), `Squeeze_Excite_Block`, `ASPP`, `AttentionBlock`.

### Data pipeline

- **Preprocessing** (`preprocess.py`): Expects input directories with `sat/` and `map/` subdirectories. Crops source images (default 1500×1500) into overlapping tiles (default 224×224) saved to `input_crop/` and `mask_crop/` under the paths specified in config (`train`/`valid` keys).
- **Data loading** (`dataset/dataloader.py`): `ImageDataset` reads from `mask_crop/` and resolves corresponding images in `input_crop/` by globbing. Returns dicts with `sat_img` and `map_img` keys. `ToTensorTarget` converts numpy arrays to tensors (HWC→CHW for images, adds channel dim to masks).
- **Augmentation** (`utils/augmentation.py`): `RescaleTarget`, `RandomRotationTarget`, `RandomCropTarget` operating on the `{sat_img, map_img}` dict format. Not currently wired into the training pipeline.

### Training loop (`train.py`)

- Uses `BCEDiceLoss` (binary cross-entropy + dice loss)
- Adam optimizer with StepLR scheduler (step=20, gamma=0.1)
- Tracks Dice coefficient and loss via `MetricTracker` running averages
- Validates and saves checkpoints every `validation_interval` steps
- Logs to TensorBoard via `MyWriter` (wraps `tensorboardX.SummaryWriter`)

### Inference (`inference.py`)

Newer, more polished script (untracked in git). Supports single-image and batch-directory modes with automatic tiling for images larger than `CROP_SIZE`, overlapping tile fusion, and optional probability map output alongside binary masks.

### Configuration (`configs/default.yaml`)

All paths and hyperparameters live here. `HParam` (`utils/hparams.py`) loads YAML into a dot-accessible dict. Key fields: `train`, `valid` (data paths), `batch_size`, `lr`, `RESNET_PLUS_PLUS` (model selection), `CROP_SIZE`, `logging_step`, `validation_interval`.

### Known issues

- `UNet.forward` uses `input` instead of `x`, so the model is broken at runtime. `UNetSmall` works correctly.
- `check_files_statistics.py` has a broken import (`import hparams as hp`) and is likely dead code.
- The `.gitignore` contains stale entries from a TTS project (e.g., `duration_modeling`, `pitch_predictor`, `rest_tts.py`) that don't exist in this repo.
- Requirements pin old versions (torch 1.6.0, numpy 1.16.3).
