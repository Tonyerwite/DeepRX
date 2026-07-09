# DeepRx Reproduction

PyTorch reproduction scaffold for **"DeepRx: Fully Convolutional Deep Learning Receiver"** by Honkala, Korpi, and Huttunen.

This project is the cleaned official-chain reproduction for **"DeepRx: Fully Convolutional Deep Learning Receiver"** by Honkala, Korpi, and Huttunen.

## What Is Implemented

- MathWorks-compatible PyTorch DeepRx network: 11 preactivation ResNet blocks, depthwise separable convolutions, dilation schedule `(1,1), (2,3), (3,6)`, and exactly 1,232,516 trainable parameters for the 16QAM/four-bit model.
- Paper/MathWorks input construction: `rxGrid + DM-RS grid + raw LS channel estimate`, converted from complex values into 10 real-valued channels for `Nrx=2`.
- QPSK/16QAM/64QAM/256QAM Gray-labelled modem with positive logits meaning bit `1`.
- Official MATLAB/5G Toolbox/6G Exploration Library bridge for PUSCH, DM-RS, LDPC, TDL/CDL channels, practical LMMSE evaluation, and PyTorch DeepRx coexecution.
- Online PyTorch training from MATLAB-generated official batches.
- Paper Fig. 6(a) reproduction script for **DeepRx vs practical LMMSE and known-channel LMMSE uncoded BER**, with 1-pilot and 2-pilot curves over `0:3:21` dB SINR.

For paper-level results, use only `scripts/train_official_matlab.py` and `scripts/reproduce_figure6a.py`. The removed pure Python runner scripts used compact channel approximations and were not the paper reproduction path.

## VSCode Setup

This folder contains a local virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For this machine's NVIDIA GPU, install the CUDA 12.8 PyTorch wheel:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
```

Install the MATLAB Engine into the same venv from the local R2025b install:

```powershell
.\.venv\Scripts\python.exe -m pip install "D:\Program Files\MATLAB\R2025b\extern\engines\python"
```

In VSCode, choose interpreter:

```text
C:\Users\Chen Yan\Desktop\AI receiver\论文复现\DeepRx\.venv\Scripts\python.exe
```

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected result: `20 passed`.

## Official Paper Reproduction

Run a quick official-chain sanity check over one SINR point:

```powershell
.\.venv\Scripts\python.exe scripts\reproduce_figure6a.py --snr-points 0 --samples-per-point 1 --n-frames 1 --output-dir outputs\figure6a_smoke
```

Run the current full Fig. 6(a) reproduction grid:

```powershell
.\.venv\Scripts\python.exe scripts\reproduce_figure6a.py --samples-per-point 500 --n-frames 1 --output-dir outputs\figure6a
```

Outputs:

- `outputs/figure6a/figure6a_metrics.json`
- `outputs/figure6a/figure6a_uncoded_ber.png`

The default checkpoint is the official MathWorks PyTorch `deeprx_30k.pth`. To evaluate a model trained by this project, pass the MATLAB-loadable state dictionary:

```powershell
.\.venv\Scripts\python.exe scripts\reproduce_figure6a.py --checkpoint checkpoints\deeprx_official_matlab_state_dict.pth --samples-per-point 500 --n-frames 1 --output-dir outputs\figure6a_custom
```

## Official MATLAB-Generated Training

Train PyTorch using MATLAB/5G Toolbox generated PUSCH batches:

```powershell
.\.venv\Scripts\python.exe scripts\train_official_matlab.py --device cuda --output checkpoints\deeprx_official_matlab.pt --save-every 500 --log-every 10
```

The defaults match the paper-scale training setup: `30000` iterations, `8` MATLAB frames per step (`80` TTIs), LAMB optimizer, learning rate `1e-2`, weight decay `1e-4`, 800-step warmup, and linear decay after 30% of training. The script uses a deterministic paper dataset index: `500000` TTIs are split into `60%` training frames and `40%` validation frames, and each 10-TTI frame in an 80-TTI training step receives its own randomized channel/SNR/DM-RS parameters.

For a quick hardware sanity check, override the batch size and step count:

```powershell
.\.venv\Scripts\python.exe scripts\train_official_matlab.py --steps 1 --n-frames 1 --device cuda --output outputs\train_sanity.pt
```

The script saves both:

- `checkpoints/deeprx_official_matlab.pt`: full training checkpoint with optimizer/history
- `checkpoints/deeprx_official_matlab_state_dict.pth`: pure PyTorch state dict for MATLAB coexecution evaluation

## Important Files

- `src/deeprx/model.py`: DeepRx architecture, input construction, loss, BER.
- `src/deeprx/matlab_bridge.py`: Python wrapper around MATLAB Engine and official MathWorks helper functions.
- `src/deeprx/official_experiments.py`: Fig. 6(a) official-chain evaluation and plotting.
- `matlab/`: MATLAB runtime helpers and wrappers for official PUSCH generation/evaluation.
- `official/`: PyTorch coexecution helper files and official `deeprx_30k.pth` checkpoint.
- `scripts/train_official_matlab.py`: PyTorch training from MATLAB-generated official batches.
- `scripts/reproduce_figure6a.py`: DeepRx vs practical and known-channel LMMSE uncoded BER reproduction for paper Fig. 6(a).
- `tests/`: regression tests for the reproduction-critical behavior.
