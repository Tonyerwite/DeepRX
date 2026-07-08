# DeepRx Reproduction

PyTorch reproduction scaffold for **"DeepRx: Fully Convolutional Deep Learning Receiver"** by Honkala, Korpi, and Huttunen.

This project was built from three sources in the parent workspace:

- `DeepRx_Fully_Convolutional_Deep_Learning_Receiver.pdf`
- MathWorks AI-native fully convolutional receiver examples copied under `../参考代码/matlab`
- The PyTorch reference under `../参考代码/DeepRx-OFDM-PyTorch-main`

## What Is Implemented

- DeepRx Table I network: 11 preactivation ResNet blocks, depthwise separable convolutions, dilation schedule `(1,1), (2,3), (3,6)`, and about 1.23M trainable parameters.
- Paper/MathWorks input construction: `rxGrid + DM-RS grid + raw LS channel estimate`, converted from complex values into 10 real-valued channels for `Nrx=2`.
- QPSK/16QAM/64QAM/256QAM Gray-labelled modem with positive logits meaning bit `1`.
- OFDM grid modulation/demodulation with the MathWorks default tensor size `[312, 14, 10] -> [312, 14, 4]` for 16QAM.
- Online data generation with randomized SNR, Doppler, channel profile, and DM-RS pattern.
- Conventional baseline: LS channel estimation, interpolation, SIMO LMMSE equalization, and max-log LLR demapping.
- Quick reproduction script that trains briefly and writes BER figures plus `metrics.json`.

The pure Python channel models are compact TDL/CDL approximations. They preserve the paper workflow and tensor interfaces, but they are not a bit-exact replacement for MATLAB 5G Toolbox / 6G Exploration Library CDL, LDPC, and PUSCH processing. Use `scripts/train_paper_config.py` for long 312-subcarrier training; exact paper curves require long training and a standards-grade channel/coding stack.

## VSCode Setup

This folder contains a local virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For this machine's NVIDIA GPU, install the CUDA 12.8 PyTorch wheel:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
```

In VSCode, choose interpreter:

```text
C:\Users\Chen Yan\Desktop\AI receiver\论文复现\DeepRx\.venv\Scripts\python.exe
```

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected result: `14 passed`.

## Quick Reproduction

```powershell
.\.venv\Scripts\python.exe scripts\quick_reproduce.py --output-dir outputs\quick_reproduction --train-steps 3 --eval-batches 2 --batch-size 2 --device cuda
```

Outputs:

- `outputs/quick_reproduction/metrics.json`
- `outputs/quick_reproduction/ber_vs_snr.png`
- `outputs/quick_reproduction/ber_vs_doppler.png`

By default the SNR grid is `0:2:12` dB. This command is a smoke test. With only a few training steps, DeepRx should not be expected to beat LMMSE; the output file marks this as `mode: smoke`.

## Paper-Size Training

This uses the 312-subcarrier, 14-symbol, 2-RX-antenna tensor dimensions and the paper-style online generator:

```powershell
.\.venv\Scripts\python.exe scripts\train_paper_config.py --steps 30000 --batch-size 20 --device cuda --save-every 1000 --log-every 10 --output checkpoints\deeprx_paper_config.pt
```

For CPU-only testing, reduce the step count:

```powershell
.\.venv\Scripts\python.exe scripts\train_paper_config.py --steps 10 --batch-size 1 --device cpu
```

The default `--max-bits 4` matches the MathWorks 16QAM PyTorch example output shape `[312,14,4]`. Use `--max-bits 8` for the paper/reference-code multi-modulation masking setup.

The script stores model, optimizer, args, and logged history in the checkpoint, so a long run leaves recoverable progress every `--save-every` steps.

After training, evaluate a checkpoint at `0:2:12` dB:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_checkpoint.py --checkpoint checkpoints\deeprx_paper_config.pt --output-dir outputs\checkpoint_eval --device cuda --snr-start 0 --snr-stop 12 --snr-step 2 --eval-samples 20 --max-bits 4
```

Outputs:

- `outputs/checkpoint_eval/checkpoint_metrics.json`
- `outputs/checkpoint_eval/checkpoint_ber_vs_snr.png`

## Important Files

- `src/deeprx/model.py`: DeepRx architecture, input construction, loss, BER.
- `src/deeprx/ofdm.py`: OFDM front end and compact fading channel models.
- `src/deeprx/data.py`: online training/evaluation sample generator.
- `src/deeprx/receiver.py`: LMMSE baseline.
- `src/deeprx/experiments.py`: quick reproduction and plotting.
- `scripts/evaluate_checkpoint.py`: checkpoint evaluation over `0:2:12` dB by default.
- `tests/`: regression tests for the reproduction-critical behavior.
