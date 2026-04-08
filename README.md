# dicom-motion-correction-mvp

> **Status: paused study artifact.** See [`DOC/05_STATUS.md`](DOC/05_STATUS.md) for why.

Rigid 2D inter-slice motion correction for CBCT volumes, targeting the Teeth FOV (5×5 cm, 0.07 mm isotropic voxel, ~720 slices). GPU-accelerated phase correlation (CuPy FFT) for coarse translation, followed by SimpleITK Mattes Mutual Information rigid refinement.

This is a research/study MVP, not a medical device. It is **not** FDA/ANVISA certified and is **not** intended for diagnostic use.

## Why this exists

CBCT scans suffer from inter-slice misalignment caused by patient micro-motion during acquisition (breathing, small head movements). At 0.07 mm voxel size, sub-millimeter motion is enough to degrade diagnostic value. The project explores whether a lightweight rigid 2D registration pipeline can recover this motion using commodity GPU hardware.

## What works

- **MVP pipeline** end-to-end in code: load DICOM series → build reference from central slices → body mask (Otsu) → phase correlation (GPU FFT) → SimpleITK Mattes MI rigid refinement → safety gate (reject/skip/apply) → resample → save new DICOM series with new UIDs.
- **Synthetic phantom validation:** on a 128³ CBCT-like phantom with 5 random injected motions (±8 px translation, ±3° rotation), the pipeline recovers:
  - mean translation error: **0.87 px**
  - max translation error: 2.66 px
  - mean rotation error: **0.52°**
  - max rotation error: 1.72°
- **GPU phase correlation alone** recovers translations to **< 0.2 px** error.
- **Real DICOM header inspection** confirmed on a Teeth 0.07 mm scan (720×720×720, uncompressed, axial, 12 bits stored in 16 allocated, isocenter ≈ image center).

## What does not work / was not finished

- **Full-volume real DICOM run** was never completed end-to-end. Blocked by the performance ceiling below.
- `metrics.py` and `report.py` were never written.
- `main.py` CLI is partially implemented.
- The `save_corrected_series` pixel clipping does not account for `BitsStored < BitsAllocated` (known TODO, deliberately left unfixed).

## Why it is paused

Two independent blockers:

1. **CPU-bound performance ceiling.** SimpleITK Mattes MI registration is CPU-only; there is no GPU path in the Python binding, and the underlying ITK 2D rigid filters have no CUDA backend. On the target hardware (Intel i7-7700, 16 GB DDR4), per-slice refinement is ~428 ms (82% of per-slice wall time). Projected full runs: Teeth ~6 min, Arch ~15–25 min, Full ~30–45 min, with 100% CPU throughout. GPU utilization sits at ~2% because only the FFT (phase correlation) runs there. Moving the MI refinement to GPU would require either rewriting it in CuPy (weeks of work, high risk of convention bugs) or switching to a deep-learning registration model (new project).

2. **Regulatory path (ANVISA).** Motion correction of clinical CBCT for diagnostic use classifies the tool as Software as a Medical Device (SaMD, likely Class II under RDC 657/2022). Public clinical use would require ISO 13485 QMS, clinical validation, ANVISA registration, and liability insurance — out of scope for a solo study.

See [`DOC/05_STATUS.md`](DOC/05_STATUS.md) for the full pause rationale and three realistic resume paths.

## Repository layout

```
dicom_motion_correction/      source package
├── config.py                 TeethProfile + ProductionTeethProfile
├── gpu_backend.py            CuPy / numpy backend with fallback
├── dicom_io.py               load + save DICOM series, preserve metadata, new UIDs
├── synthetic.py              CBCT-like phantom + motion injection + recovery error
├── registration.py           phase correlation + SimpleITK MI rigid refinement
└── main.py                   CLI (partial)

DOC/                          architecture and decision log
├── 01_PURPOSE.md             scope + safety contract
├── 02_STACK.md               versions, paths, install gotchas
├── 03_ARCHITECTURE_NOW.md    module map, data flow, sign conventions
├── 04_ALGORITHM.md           pipeline math, parameter table, validation strategy
├── 05_STATUS.md              ← primary doc: current state + pause rationale
├── 06_HEXAGONAL_TARGET.md    target architecture (CANCELLED, historical)
├── 07_PHYSICS_AND_MOTION_MODEL.md  clinical priors (partially verified)
└── 08_NEXT_REAL_DICOM_TEST.md      real DICOM test plan (partially executed)
```

## How to run (if you want to reproduce)

Full setup details are in [`DOC/02_STACK.md`](DOC/02_STACK.md). Short version:

```bash
# Create venv with Python 3.11
python -m venv venv
venv/Scripts/activate       # or source venv/bin/activate on Unix

# Install dependencies
pip install numpy scipy pydicom SimpleITK matplotlib tqdm pynvml psutil
pip install cupy-cuda12x    # GPU, optional
pip install nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-nvjitlink-cu12

# Run synthetic phantom validation (no GPU required, no DICOM required)
python -m dicom_motion_correction.synthetic
python -m dicom_motion_correction.registration
```

The synthetic test is the only fully working end-to-end entry point. It generates a CBCT-like phantom, injects known motion, runs the full pipeline, and prints recovery errors.

## Sign conventions

Multiple sign conventions are in play (phase correlation vs. SimpleITK Euler2D vs. `InjectedMotion` vs. `SliceCorrection`). They are documented at the top of each module that touches them and in [`DOC/03_ARCHITECTURE_NOW.md`](DOC/03_ARCHITECTURE_NOW.md). Mixing them without reading those headers is the #1 cause of bugs in this codebase (see B1 in `DOC/05_STATUS.md`).

## License

Licensed under the **Apache License, Version 2.0**. See [`LICENSE`](LICENSE) for the full text.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## ⚠️ Medical Disclaimer

This software is for **research and educational purposes only**.
It is **NOT** intended for clinical use, diagnosis, or treatment.

The authors assume no responsibility for any use of this code in medical or clinical environments.

By using this software, you agree that you are solely responsible for ensuring compliance with any applicable laws or regulations in your jurisdiction, including but not limited to medical device regulations (ANVISA, FDA, CE-MDR, PMDA, and equivalents).

## Acknowledgements

Co-authored with Claude (Anthropic) as a sounding board for idea validation, architectural review, and sanity-checking technical decisions. All scoping, priorities, and the decision to pause were made by the human author.
