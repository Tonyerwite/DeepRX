# Figure 6(a) Whole-Band SNR Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lossless SNR-domain conversion and redraw the completed MC=500 Figure 6(a) data on the paper's whole-band SNR axis without rerunning MATLAB.

**Architecture:** Keep evaluation and raw metrics unchanged. Add pure conversion and artifact-writing functions beside the existing plotting code, expose them through a small replot CLI, and publish separate corrected assets so the source run remains immutable.

**Tech Stack:** Python 3.9, matplotlib, pytest, JSON, Pillow for image verification.

## Global Constraints

- Use Python exclusively for drawing and visual QA.
- Preserve every existing BER value exactly.
- Preserve the original MC=500 JSON and PNG byte-for-byte.
- Use `paper_whole_band_snr_db = matlab_per_re_snr_db - 10*log10(512/312)`.
- Write corrected outputs under new filenames.

---

### Task 1: SNR Conversion And Corrected Metrics

**Files:**
- Modify: `tests/test_official_experiments.py`
- Modify: `src/deeprx/official_experiments.py`

**Interfaces:**
- Produces: `figure6a_snr_offset_db() -> float`
- Produces: `convert_figure6a_metrics_to_paper_snr(metrics: Dict) -> Dict`

- [ ] **Step 1: Write failing conversion tests**

Add tests asserting an offset of `2.15115366957388`, corrected endpoints of `-2.15115366957388` and `18.84884633042612`, both SNR arrays in the returned object, and exact equality of all curve arrays.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_official_experiments.py -q`

Expected: import or attribute failure because the conversion functions do not exist.

- [ ] **Step 3: Implement the pure conversion functions**

Use `copy.deepcopy`, constants `512` and `26*12`, and conversion metadata containing the formula and offset. Leave the legacy `snr_db` field unchanged.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_official_experiments.py -q`

Expected: all focused tests pass.

### Task 2: Replot Artifact Writer And CLI

**Files:**
- Modify: `tests/test_official_experiments.py`
- Modify: `src/deeprx/official_experiments.py`
- Create: `scripts/replot_figure6a.py`

**Interfaces:**
- Extend: `plot_figure6a(metrics: Dict, path: Path, *, snr_domain: str = "matlab_per_re") -> None`
- Produces: `write_paper_snr_figure6a_artifacts(source_metrics_path: Path, corrected_metrics_path: Path, corrected_figure_path: Path) -> Dict`
- CLI: `python scripts/replot_figure6a.py --metrics SOURCE [--output-metrics PATH] [--output-figure PATH]`

- [ ] **Step 1: Write failing artifact tests**

Create temporary source metrics, call the artifact writer, and assert that the source bytes are unchanged, both corrected files exist, the PNG is nonempty, and the corrected JSON contains the paper-axis array.

- [ ] **Step 2: Run the artifact test and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_official_experiments.py -q`

Expected: failure because the writer and paper-axis plotting mode do not exist.

- [ ] **Step 3: Implement plotting mode, writer, and CLI**

For `paper_whole_band`, convert metrics and plot `paper_whole_band_snr_db` with xlabel `Whole-band SINR (dB)`. Reject unsupported domains. The writer must reject output metrics paths equal to the source path and use the existing atomic JSON writer.

- [ ] **Step 4: Run focused and full tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_official_experiments.py -q`

Run: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: all tests pass.

### Task 3: Generate And Publish Corrected Assets

**Files:**
- Create: `outputs/final_reproduction_30k_mc500/figure6a/figure6a_metrics_paper_snr.json`
- Create: `outputs/final_reproduction_30k_mc500/figure6a/figure6a_uncoded_ber_paper_snr.png`
- Create: `docs/assets/figure6a_30k_mc500_metrics_paper_snr.json`
- Create: `docs/assets/figure6a_30k_mc500_paper_snr.png`
- Modify: `README.md`

**Interfaces:**
- Consumes: completed raw metrics at `outputs/final_reproduction_30k_mc500/figure6a/figure6a_metrics.json`
- Produces: separate corrected JSON and PNG artifacts.

- [ ] **Step 1: Record source hashes**

Expected SHA-256 values:

- Metrics: `E162DFFDE99E5BBED3BE186F51E788417190CEDCF1907B76817EA70D2981A61A`
- PNG: `8DB77C135FA43ACA240891DB32F2B434E07205CC85F1C6C70A880FEA7D725D1C`

- [ ] **Step 2: Run the replot CLI**

Run the CLI with explicit corrected output paths under the final reproduction directory, then copy those corrected artifacts into `docs/assets/` under the specified new names.

- [ ] **Step 3: Update README**

Display the corrected asset, label the table's source SNR as MathWorks per-RE SNR, document the 2.151 dB conversion, and add the replot command without removing the raw-result provenance.

- [ ] **Step 4: Verify source preservation and corrected data**

Recompute both source hashes, compare corrected curves to source curves exactly, and check the corrected SNR endpoints.

### Task 4: Visual And Final Verification

**Files:**
- Verify: `docs/assets/figure6a_30k_mc500_paper_snr.png`

**Interfaces:**
- Produces: verified, user-facing corrected figure.

- [ ] **Step 1: Run full tests and repository checks**

Run: `.\.venv\Scripts\python.exe -m pytest -q`

Run: `git diff --check`

- [ ] **Step 2: Inspect image dimensions and pixels**

Use Pillow to assert nonzero dimensions, nonblank grayscale range, and a valid PNG format.

- [ ] **Step 3: Inspect the native PNG visually**

Confirm all five curves, legend entries, log ticks, corrected x-axis label, and plot boundaries are legible with no overlaps.

- [ ] **Step 4: Commit the implementation**

Commit source, tests, README, plan, and corrected documentation assets. Keep ignored run outputs on disk without forcing them into Git.
