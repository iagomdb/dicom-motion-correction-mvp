# Status

**PROJECT STATUS: PAUSED on 2026-04-08.**

## Why paused

Two independent blockers, either of which is sufficient:

1. **CPU-bound performance ceiling.** SimpleITK Mattes MI registration is CPU-only by design (no GPU path exists in the Python binding, and the underlying ITK 2D rigid filters do not have a CUDA backend). Benchmark on the target i7-7700 shows the `register_slice` MI refinement taking ~428 ms per slice (82% of the per-slice wall time) out of ~521 ms total. Extrapolated full-volume times on available hardware:
   - Teeth 720 slices: ~6 min (acceptable but already uses 100% CPU)
   - Arch 1024 slices, 1000×1000: ~15–25 min (projected)
   - Full 1500×1000×720: ~30–45 min (projected)
   A GPU-native path would require either (a) rewriting MI + optimizer in CuPy (~weeks, high risk of sign/convention bugs like B1), or (b) switching to a deep-learning registration model such as VoxelMorph / TransMorph (new project: dataset + training + PyTorch stack). Neither is a small extension of this MVP.

2. **Regulatory path (ANVISA).** Motion correction of clinical CBCT for diagnostic use classifies the tool as Software as a Medical Device (SaMD, likely Class II under RDC 657/2022). Public distribution or clinical use would require ISO 13485 QMS, clinical validation, ANVISA registration, and liability insurance — investment that does not fit the scope of a solo research MVP. In addition, the project implicitly exposes structural issues of a specific commercial CBCT system, creating unwanted reputational and legal exposure for the author (ex-member of that vendor's engineering team).

## State at pause

**Working:**
- MVP package skeleton (`config.py`, `gpu_backend.py`, `dicom_io.py`, `synthetic.py`, `registration.py`)
- Phase correlation (GPU FFT via CuPy) recovers injected translation to <0.2 px on synthetic phantom
- B1 resolved: SimpleITK MI rigid refinement converging correctly after running in pixel units, dropping mask from phase-corr, and using `CenteredTransformInitializer`. Synthetic phantom 128³ with 5 random motions: mean translation error 0.87 px, mean rotation error 0.52°.
- Real DICOM inspection completed: target Teeth scan is 720 slices × 720×720, 0.07 mm isotropic, uncompressed, axial orientation `[1,0,0,0,1,0]`, isocenter at ~image center. Pixel format confirmed: **12 bits stored in 16 allocated, PixelRepresentation signed** — matches user's prior expertise from CBCT bit-packing work. `dicom_io.save_corrected_series` current clipping logic (`np.iinfo(int16)` full range) is therefore technically incorrect for this vendor but not catastrophic; documented as TODO, not fixed.
- Production benchmark completed for baseline `TeethProfile` (see section below).

**Not completed:**
- `main.py` CLI was partially implemented by parallel agent (see `dicom_motion_correction/main.py` if present) but the full real-DICOM end-to-end run was never executed successfully due to the performance ceiling.
- `ProductionTeethProfile` was added to `config.py` with `pyramid_shrink=[8,4,2]`, reduced iterations, reduced sampling. It failed the synthetic rotation acceptance criterion (1.61° vs 1.0° target) at 128×128 phantom size — diagnosed by architect as artifact of phantom being too small for aggressive pyramid, not a defect of the profile itself. Validation on 512×512 phantom never run.
- `metrics.py` and `report.py` never written.
- Bit-packing clipping fix in `save_corrected_series` never applied.

## If resumed in the future

The realistic path forward is not to continue optimizing this SimpleITK-based pipeline. It is to either:
- (a) Start a new project using a deep-learning registration model (VoxelMorph / TransMorph / custom) with the synthetic phantom and injection machinery from `synthetic.py` reused as training/validation data generator, **or**
- (b) Run the current pipeline on modern CPU hardware (something 2020+, 8+ real cores, DDR5) as-is, treating the slow runtime as acceptable for internal / one-off use, **or**
- (c) Abandon the correction direction entirely and instead produce a motion detection/quality report (same phantom, same phase correlation, same metrics, no resampling or DICOM writeback) — this sidesteps the SaMD classification since no reprocessed image is produced, only a quality flag.

All three are separate projects. This repository is a frozen study artifact and should be treated as reference material, not as a base to extend in place.

---

## Historical log (kept for reference)

### Active work (at time of pause)

- [done] registration.py — B1 fixed: run SimpleITK in pixel units, drop mask from phase-corr, use CenteredTransformInitializer, re-verify signs. Mean translation error 0.87 px, max 2.66 px, mean rotation error 0.52°, max 1.72° on the 5-slice synthetic case.
- [paused] main.py + first real DICOM test — blocked by performance ceiling described above

## Production benchmark

Real Teeth scan: 720 slices, 720×720, 0.07 mm isotropic, 12 bits stored in 16, uncompressed, axial, IOP=[1,0,0,0,1,0], isocenter at (360,360) ≈ image center (359.5,359.5). Volume loaded as float32 after RescaleSlope/Intercept, dtype for save is original uint16.

### Baseline (TeethProfile, 20 slices, indices 350..369, GPU backend)
- Total time: 10.42 s
- Per-slice avg: 0.521 s
  - phase_corr: 72.2 ms/slice
  - sitk.Execute: 427.8 ms/slice (82% of per-slice time)
  - other overhead: ~21 ms
- RSS peak: 3.34 GB (volume-load dominated: 3.11 GB right after load)
- VRAM peak: 1135 MB / 6442 MB total
- Extrapolated full run (711 non-ref slices): **~6.2 min**

SimpleITK MI refinement is the bottleneck. phase_corr is cheap. GPU is only used by phase_corr (cufft) — MI runs on CPU threads (sitk default 8 threads = all cores).

### Step 3 — ProductionTeethProfile synthetic validation: FAILED rotation criterion

`ProductionTeethProfile` (pyramid_shrink=[8,4,2], pyramid_sigma=[4,2,1], optimizer_iterations=40, metric_sampling_percentage=0.15) on the 128×128×32 synthetic phantom, 5 random motions, seed 7:

```
mean_translation_error_px: 0.190     (target < 2.0 — PASS)
max_translation_error_px:  0.454
mean_rotation_error_deg:   1.607     (target < 1.0 — FAIL)
max_rotation_error_deg:    3.265
```

Translation recovery is actually **better** than baseline TeethProfile (0.19 px vs 0.87 px) because the coarser pyramid + less sampling avoids the translation plateau seen in B1. Rotation recovery is worse: the aggressive shrink [8,4,2] plus sigma 4.0 at level 0 washes out the rotational gradient of the small 128×128 phantom — the finest level in this profile is 64×64 equivalent, which leaves few gradient features to constrain rotation.

**Hypothesis:** the failure is phantom-specific. At 128×128 a shrink of 8 leaves 16×16 at the coarsest level — MI has almost nothing to optimize against. On the real 720×720 scan the coarsest level is 90×90, which should behave much better. But this is speculation, not evidence.

**Blocking per instructions:** criterion mean_rotation_error_deg < 1° was not met. Stopping before Step 4. Needs the architect to decide one of:

1. Relax the synthetic acceptance criterion to < 2° (real FOV has 32× more pixels; the synthetic phantom may not be representative of production workload, and the *real* rotation accuracy requirement is whatever still registers teeth correctly).
2. Soften the production profile to `pyramid_shrink=[4,2,1]` and keep only the iteration/sampling reductions (probably ~40% speedup instead of target 3×).
3. Keep `pyramid_shrink=[8,4,2]` but bump optimizer_iterations back to 60 or raise sampling to 0.25 — may recover rotation accuracy on the phantom at cost of speed.
4. Split profiles: validate production profile directly on the real DICOM (no synthetic gate), accept the risk.

Awaiting decision before proceeding to Steps 4/5.


## Working

| Component | Evidence |
|---|---|
| Python venv + lightweight deps | `python -m dicom_motion_correction.gpu_backend` prints `GPU: NVIDIA GeForce RTX 2060 (6144 MB, 5095 MB free)` |
| `config.py` / `TeethProfile` | imports clean, defaults match `04_ALGORITHM.md` table |
| `gpu_backend.py` | GPU detected, fallback to CPU works via `force_cpu=True` |
| `dicom_io.py` | written, **not yet exercised** on real or fake DICOM |
| `synthetic.py` phantom + injection | `python -m dicom_motion_correction.synthetic` produces `D:/dicom_mc/cache/synthetic_demo.png` |
| `registration.phase_correlation_translation` | recovers injected translation to **<0.2 px** on 5 random motions in 128×128×32 phantom (CPU backend, FFT path) |
| CuPy GPU JIT (NVRTC) | `cp.arange(1M).sum()` works after installing `nvidia-cuda-nvrtc-cu12` + `nvidia-nvjitlink-cu12` |

## Broken / blocked

### B1. [RESOLVED 2026-04-08] SimpleITK rigid refinement degrades a good phase-correlation init
**Severity:** high. Pipeline is unusable until fixed.

**Resolution summary (2026-04-08):** Three independent bugs stacked on top of each other:
1. **Mask destroyed the phase-correlation signal.** `phase_correlation_translation` was being called with the body mask, which multiplies ref and mov by a hard binary mask. The mask boundary creates a strong step discontinuity that dominates the cross-power spectrum and pulls the peak to (0,0). Fixed by not passing the mask to phase-corr (it is still used by the MI refinement via `SetMetricFixedMask`).
2. **SimpleITK optimizer was mis-scaled by physical (mm) units at 0.07 mm/px.** `RegularStepGradientDescent(learningRate=1.0, minStep=1e-4)` combined with `SetOptimizerScalesFromPhysicalShift()` and a 0.07 mm spacing meant the learning rate was effectively ~14 px and the min step ~0.0014 px — overshoot and never converge. Fixed by running the registration in pixel units (`spacing=(1.0, 1.0)` on both fixed and moving images). Physical spacing has no effect on a purely in-slice rigid registration; the conversion to mm only matters for the safety-gate thresholds, which already live outside the SimpleITK call.
3. **Manual rotation center arithmetic was a latent bug.** Replaced with `sitk.CenteredTransformInitializer(..., GEOMETRY)`, which eliminates the hypothesis independently of whether it was firing.

**Sign convention (re-verified empirically, see `register_slice` comments):**
- `phase_correlation_translation` returns `-injected_shift` (i.e. the shift that aligns moving back to reference).
- SimpleITK Euler2D translation converges to `+injected_shift` in pixel units, so the phase-corr estimate is *negated* when used as init.
- `SliceCorrection.translation_x/y` is reported as `-Euler2D.translation` so that `injected + recovered ≈ 0`.
- `SliceCorrection.rotation_deg` is reported as `+Euler2D.angle` (NOT negated) — the angle comes out already in the "motion removed" sign due to the row/col vs x/y axis flip between synthetic.inject_motion and SimpleITK's in-plane convention.

**Evidence after fix** (`python -m dicom_motion_correction.registration`, CPU backend, 128×128×32 phantom, 5 random motions, seed 7):
```
mean_translation_error_px: 0.87
max_translation_error_px: 2.66
mean_rotation_error_deg: 0.52
max_rotation_error_deg: 1.72
```
Down from mean 23 px / max 51 px / rotation up to 9°. All 5 slices now pass the safety gate. Two of the five slices (z=23, z=28) still have ~2–3 px translation residual — not a structural bug, likely just the MI optimizer plateauing on a feature-poor phantom. Acceptable for MVP; revisit if real-DICOM evaluation shows similar residuals.

---

**Original report follows (kept for history):**


**Symptom:** On the synthetic 128×128×32 phantom, phase correlation alone gives sub-pixel coarse estimates (verified, see "Working"). After feeding those as the Euler2D init and running Mattes MI, the final transform diverges. Mean translation error jumps from <0.2 px to **23 px**, max to **51 px**. 2 of 5 slices end up rejected by the safety gate.

**Evidence (from `python -m dicom_motion_correction.registration`, CPU backend):**
```
z=21  injected=(-7.92,+5.14,+2.24°)  coarse=(+8.06,-5.10)  final=(+8.50,-5.33,+0.79°)   ok-ish but rotation lost
z=22  injected=(-0.51,-3.15,+1.78°)  coarse=(+0.81,+3.13)  final=(+1.96,+17.27,+4.70°)  ty diverged 14 px
z=23  injected=(-3.92,-0.88,-1.33°)  coarse=(+3.85,+0.92)  final=(-6.87,+8.63,-0.89°)   sign of tx flipped
z=26  injected=(+0.86,+7.93,+0.03°)  coarse=(-1.00,-7.96)  final=(-10.60,+43.08,-8.75°) rejected
z=28  injected=(+1.95,+7.82,+1.76°)  coarse=(-1.92,-7.86)  final=(-24.54,-47.16,+1.69°) rejected
```

**Hypotheses, ranked by likelihood:**
1. **Sign/axis confusion at the phase-corr → SimpleITK boundary.** `register_slice` lines ~140–160 negate the phase-corr translation to convert from "align moving to ref" into SimpleITK's "fixed → moving" convention, then convert px → physical units via `spacing[0/1]`. SimpleITK uses `(x, y)` order; numpy/scipy/`ndimage` use `(row, col) = (y, x)`. Phase correlation returns `(tx_columns, ty_rows)` per its docstring. Either the negation is double-counted or x/y is swapped on input or output.
2. **Center of `Euler2DTransform` may be wrong.** It's set from `origin + spacing*(size-1)/2` per axis using `size = reference_sitk.GetSize()`. SimpleITK `GetSize()` returns `(W, H)` (x then y). Need to verify the indexing matches the spacing indexing.
3. **Synthetic phantom too feature-poor.** Mattes MI may have weak gradients far from the optimum. Less likely because the init is already very close (<1 px) and MI should still descend, not climb.
4. **`SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()` interaction with `SetOptimizerScalesFromPhysicalShift()`.** Mixing voxel-unit smoothing with physical-unit scales can confuse the optimizer in non-isotropic spacing. Phantom is isotropic 1.0, so probably not it, but worth ruling out.

**Reproduction:**
```
cd "c:/Users/iago/Desktop/Geral/VS CODE/Dicom"
"D:/dicom_mc/venv/Scripts/python.exe" -m dicom_motion_correction.registration
```
Uses `force_cpu=True` so it does not depend on CuPy/NVRTC state.

**Diagnostic suggestions for whoever fixes this:**
- Hard-code an injected motion (e.g. tx=+5, ty=0, rot=0), feed phase-corr init manually, print SimpleITK's metric value at init AND after a single iteration. If metric *worsens* after iteration, the optimizer is descending in the wrong space — sign/axis bug confirmed.
- Bypass phase-corr init (set Euler2D translation=(0,0)), let SimpleITK find it from scratch. If it converges correctly, the bug is at the boundary, not in the SimpleITK setup.
- Try `centeredTransformInitializer` (`sitk.CenteredTransformInitializer`) instead of manual center calc — eliminates B1.2 as a hypothesis.

### B2. NVIDIA cublas/cusolver/cusparse/curand wheels not installed
**Severity:** medium. GPU code paths that use these libs will crash. FFT (cufft) is installed, so phase correlation works on GPU. Random generation (`cp.random.*`) and linear algebra do not.
**Symptom:** `pip install nvidia-cublas-cu12` fails because its build pulls `wheel-stub` from pypi and the connection times out.
**Workaround:** `pip install --no-build-isolation nvidia-curand-cu12 nvidia-cublas-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 --cache-dir D:\dicom_mc\cache --timeout 600`. Not yet attempted.
**Impact on phase 1:** registration only needs cufft (installed). cuRAND only matters if/when we use `cp.random` for synthetic data on GPU. Currently synthetic uses numpy. Not blocking the fix for B1.

## Not started

- `metrics.py` (quality report dataclass, edge sharpness)
- `report.py` (PNG with 4-panel before/after/displacement/table)
- `main.py` (CLI, arg parsing, terminal output)
- Test on real DICOM Teeth 0.07 mm series (no test data provided yet)
- Hexagonal refactor (see `06_HEXAGONAL_TARGET.md`)
