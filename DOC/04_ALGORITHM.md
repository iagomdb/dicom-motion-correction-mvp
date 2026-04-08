# Algorithm

## Pipeline (per slice)

1. **Coarse translation via phase correlation** (GPU FFT)
   - `R = F(ref) · conj(F(mov)) / |·|`
   - `r = ifft(R).real`
   - Argmax → integer peak. Wraparound: if peak > N/2, peak -= N.
   - Sub-pixel: 3-point parabolic fit on the peak's row/column.
   - Output: `(tx, ty)` in pixels, sign = "shift to apply to moving so it aligns to reference".

2. **Rigid refinement via SimpleITK Mattes MI** (CPU)
   - Transform: `Euler2DTransform` (3 DOF: angle, tx, ty in physical units)
   - Initialized with phase-corr translation (negated; convention conversion)
   - Centered at the physical center of the fixed image
   - Metric: `MattesMutualInformation`, 50 histogram bins, REGULAR sampling 50%
   - Optimizer: `RegularStepGradientDescent`, lr=1.0, minStep=1e-4, 100 iter, gradientTol=1e-6
   - Pyramid: shrink `[4, 2, 1]`, smoothing sigmas `[2, 1, 0]` (in voxel units, not physical)
   - Scales: `SetOptimizerScalesFromPhysicalShift()`
   - Mask: body mask of reference, set as `MetricFixedMask`

3. **Safety gating** — applied to the final transform parameters, NOT clamped:
   - **Reject** if `|rot| > 5°` or `|tx| > max_translation_mm/spacing` or `|ty| > max_translation_mm/spacing` → return original slice unchanged, mark `rejected_reason="exceeds_limits"`.
   - **Skip** if `|tx| < motion_threshold_mm/spacing` AND `|ty| < threshold` AND `|rot| < 0.1°` → return original slice, mark `rejected_reason="below_threshold"`.
   - Otherwise: resample with the final transform, mark `was_corrected=True`.

4. **Reference handling**: the central `k=9` slices used to build the reference are not registered. They pass through unchanged with `rejected_reason="reference"`.

## Reference construction

`reference = mean(volume[mid - k//2 : mid + k//2 + 1])` with k=9 (configurable in profile). Mean reduces noise. One reference for the whole volume in this phase. Adaptive bands (multiple references along Z) are deferred to multi-FOV phase.

## Body mask

Otsu threshold on Gaussian-smoothed reference, binary closing (3 iter), keep largest connected component. For Teeth FOV the body fills most of the frame so the mask is mostly 1s — its main job is to suppress air-region bias in the Mattes MI metric.

## Parameter table (Teeth 0.07 mm profile)

| Param | Value | Why |
|---|---|---|
| `motion_threshold_mm` | 0.05 | ~0.7 px @ 0.07 mm; below this is sub-pixel noise |
| `max_translation_mm` | 1.4 | ~20 px; gross motion is unrecoverable, reject |
| `max_rotation_deg` | 5.0 | Above this, rigid 2D is no longer the right model |
| `reference_method` | `mean_central_k` | Robust against single-slice noise |
| `reference_k` | 9 | ~0.6 mm thick reference, anatomically coherent |
| `pyramid_shrink` | `[4, 2, 1]` | Coarse-to-fine, escapes local minima |
| `pyramid_sigma` | `[2.0, 1.0, 0.0]` | Matches shrink levels |
| `mi_histogram_bins` | 50 | Standard for MI on medical images |
| `optimizer_iterations` | 100 (per level) | Enough for convergence in 2D rigid |
| `roi_mask` | `auto` | Otsu-based body mask |

All thresholds are stored in **mm** in the profile and converted to **px** at use-site via `PixelSpacing`. This makes the same profile valid across voxel sizes (relevant when other FOVs come online).

## Validation strategy

Synthetic phantom with **known injected motion** is the only way to measure registration accuracy quantitatively. `synthetic.compute_recovery_error` returns mean/max translation error in px and rotation error in deg. Acceptance: **mean translation error < 0.1 px, mean rotation error < 0.1°** on the default phantom with motions up to ±8 px / ±3°.

Phase correlation alone currently meets this bar. The full pipeline (with SimpleITK refinement) does not — see `05_STATUS.md`.
