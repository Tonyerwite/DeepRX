# DeepRx Reproduction Prerequisite Audit

Date: 2026-07-08

## Environment

- MATLAB R2025b executable: `D:/Program Files/MATLAB/R2025b/bin/matlab.exe`
- MATLAB release/version: `R2025b`, `25.2.0.2998904`
- Python venv: `C:/Users/Chen Yan/Desktop/AI receiver/论文复现/DeepRx/.venv/Scripts/python.exe`
- PyTorch CUDA: `2.8.0+cu128`, CUDA runtime `12.8`
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU

## MATLAB Toolboxes

Verified installed/enabled in R2025b:

- 5G Toolbox 25.2
- Deep Learning Toolbox 25.2
- Parallel Computing Toolbox 25.2
- Communications Toolbox 25.2
- DSP System Toolbox 25.2
- Signal Processing Toolbox 25.2

## 6G Exploration Library

Official source checked:

- MathWorks documentation: `https://www.mathworks.com/help/5g/6g-exploration-library.html`
- MathWorks File Exchange download entry: `https://www.mathworks.com/matlabcentral/fileexchange/157771-6g-exploration-library-for-5g-toolbox`

Downloaded official signpost:

- `vendor/mathworks/6gexplorationlibrary.mlpkginstall`
- Package metadata says:
  - Repository: MathWorks
  - Name: 6G Exploration Library
  - FullName: 6G Exploration Library for 5G Toolbox
  - BaseProduct: 5G Toolbox
  - BaseCode: NRX5G

Current status:

- The signpost is downloaded.
- The actual add-on payload is not installed yet.
- `matlab.addons.install` cannot install `.mlpkginstall`; it requires `.mltbx`.
- DeepRx helper functions are not currently on MATLAB path:
  - `hCreateDeepRx`
  - `hTrainDeepRx`
  - `hGeneratePUSCHTrainingBatch`
  - `hEvaluatePUSCHAtSNR`

Action needed before exact official-data alignment:

- Open `vendor/mathworks/6gexplorationlibrary.mlpkginstall` with MATLAB R2025b Add-On Installer, sign in if prompted, and complete installation.
- After installation, rerun the helper-function path check.

## GPU And Parallel

MATLAB R2025b initial `gpuDevice` failed because RTX 5060 has compute capability 12.0, newer than the bundled CUDA libraries.

Verified fix:

```matlab
parallel.gpu.enableCUDAForwardCompatibility(true)
gpuDevice
```

After enabling forward compatibility:

- `canUseGPU` returns true
- `gpuDevice` detects NVIDIA GeForce RTX 5060 Laptop GPU
- Small GPU matrix computation succeeds

Parallel pool:

- `canUseParallelPool` returns true
- Local process pool with 2 workers started successfully
- `parfeval(@plus,1,1,2)` returned 3

## Official Example Parameters To Match

From the MathWorks AI-native fully convolutional receiver example:

- `TrainNow = false` by default
- `UseParallel = false` by default
- `LearnRate = 0.001`
- `NTrainSamples = 30e3` for the example
- `NFrames = 1` for training iteration
- `CarrierFrequency = 3.5e9`
- `NSizeGrid = 26`
- `SubcarrierSpacing = 15`
- `ModulationType = 16QAM`
- `CodeRate = 658/1024`
- Training SNR range: `[-4, 32]` dB
- Delay spread range: `[10e-9, 300e-9]`
- Doppler range: `[0, 500]` Hz
- DM-RS additional position limits: `[0, 1]`
- DM-RS configuration type limits: `[1, 2]`
- Training channels: `CDL-B`, `CDL-C`, `CDL-D`, `TDL-B`, `TDL-C`, `TDL-D`
- Validation channels: `CDL-A`, `CDL-E`, `TDL-A`, `TDL-E`
- Input size: `[312, 14, 10]`
- Output size for 16QAM: `[312, 14, 4]`
- Network: 11 ResNet blocks, 1.2325M learnables

## Current Python Alignment Status

Already aligned:

- PyTorch CUDA environment is available.
- DeepRx input channel construction uses `rxGrid + dmrsGrid + rawChanEstGrid`, then real/imag stacking.
- Default tensor dimensions support `[312, 14, 10]`.
- `--max-bits 4` supports MathWorks 16QAM output `[312, 14, 4]`.
- Network block widths and dilations match the paper table.
- Tests cover input construction, masks, BER convention, OFDM sanity, LMMSE sanity, SNR grid, and checkpoint evaluation.

Not fully aligned yet:

- Python channel/data generator is still a compact approximation, not official MATLAB PUSCH/CDL/TDL/LDPC data generation.
- Python does not yet call MATLAB/6G Library helpers to generate official training/evaluation tensors.
- Coded BER/throughput with UL-SCH decoding is not yet implemented in Python.
- Exact DM-RS configuration type/additional position behavior should be sourced from the installed 6G Exploration Library helpers once available.

Next required milestone:

- Install 6G Exploration Library.
- Export or bridge official MATLAB-generated DeepRx inputs/labels into Python.
- Replace approximate Python data generator for paper-level runs with official-generated tensors.
