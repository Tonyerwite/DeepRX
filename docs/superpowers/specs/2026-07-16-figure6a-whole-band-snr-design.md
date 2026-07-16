# Figure 6(a) Whole-Band SNR Correction Design

## Goal

Redraw the completed MC=500 DeepRx Figure 6(a) reproduction on the paper's whole-band SNR axis without rerunning MATLAB evaluation or altering any original BER values.

## Scientific Contract

- Core conclusion: the apparent common BER penalty is predominantly a 2.151 dB SNR-domain offset, not a receiver-performance regression.
- Evidence: all five receiver curves retain their measured BER values and move together on the corrected horizontal axis.
- Figure type: single-panel quantitative comparison.
- Backend: Python with the repository's existing matplotlib workflow.
- Export: PNG with the same curve styles as the existing reproduction and an explicit `Whole-band SINR (dB)` axis label.

## Conversion

The R2025b MathWorks helper interprets SNR per resource element and per antenna. The paper defines SNR over the whole OFDM bandwidth. For 26 PRBs at 15 kHz subcarrier spacing:

- Occupied subcarriers: `26 * 12 = 312`
- FFT size: `512`
- Offset: `10 * log10(512 / 312) = 2.15115366957388 dB`
- Paper-axis coordinate: `paper_whole_band_snr_db = matlab_per_re_snr_db - offset_db`

The completed MATLAB samples at `0, 3, ..., 21 dB` therefore appear at `-2.151, 0.849, ..., 18.849 dB` on the paper axis.

## Data Preservation

- Do not modify or delete `outputs/final_reproduction_30k_mc500/figure6a/figure6a_metrics.json`.
- Do not modify or delete `outputs/final_reproduction_30k_mc500/figure6a/figure6a_uncoded_ber.png`.
- Generate a separate corrected JSON containing both SNR domains, conversion metadata, and the unchanged BER arrays.
- Generate a separate corrected PNG; no MATLAB engine or model inference is used during replotting.

## Program Changes

1. Add a tested conversion helper to `src/deeprx/official_experiments.py`.
2. Make the plotter select an explicit SNR domain while retaining legacy behavior for existing callers.
3. Add `scripts/replot_figure6a.py` to load completed metrics and emit corrected JSON/PNG files.
4. Add focused tests for the exact 2.151 dB conversion, unchanged BER arrays, and non-overwriting output behavior.
5. Publish the corrected figure as a new documentation asset and update the README figure reference and SNR-definition note.

## Outputs

- `outputs/final_reproduction_30k_mc500/figure6a/figure6a_metrics_paper_snr.json`
- `outputs/final_reproduction_30k_mc500/figure6a/figure6a_uncoded_ber_paper_snr.png`
- `docs/assets/figure6a_30k_mc500_paper_snr.png`
- `docs/assets/figure6a_30k_mc500_metrics_paper_snr.json`

## Verification

- Unit tests must fail before implementation and pass afterward.
- The corrected first and last coordinates must be approximately `-2.15115367` and `18.84884633 dB`.
- Every corrected BER array must exactly equal the source BER array.
- Source metrics and PNG hashes must remain unchanged.
- The corrected PNG must be nonblank, legible, and free of overlaps at its native resolution.
