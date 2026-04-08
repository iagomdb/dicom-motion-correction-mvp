# Physics & Motion Model

**Status: working hypotheses, not verified facts.** Source: a member of the CBCT engineering team whose primary expertise is DICOM pixel encoding (bit packing, dtype, signed/unsigned). The clinical/geometry claims below are *his informed guesses*, not measurements. Treat them as the working model until confirmed against real DICOM headers and an actual physicist or clinician.

**Confirmation checklist (do this as soon as a real DICOM Teeth scan is available):**
- [ ] Read `ImagePositionPatient` of the central slice. Is `(0,0)` of LPS near the slice center? If yes, the isocenter-at-origin claim is supported.
- [ ] Read `ImageOrientationPatient`. Is it `[1,0,0, 0,1,0]` (axial, no rotation, no tilt)?
- [ ] Confirm `PixelSpacing` is isotropic (sx == sy) and matches the expected 0.07 mm.
- [ ] Spot-check 10 slices: is `ImagePositionPatient[2]` (z) monotonic and equally spaced?
- [ ] If any of the above fails, this entire document needs revision before the algorithm trusts the geometry.

## Acquisition geometry (hypothesis)

- CBCT C-arm rotates around a **fixed isocenter at physical XYZ = (0, 0, 0)** in patient coordinates (LPS, as encoded in DICOM `ImagePositionPatient`). *Plausible but unverified — confirm via header read.*
- Operator positions the C-arm so that the target tooth coincides with the isocenter before acquisition. *Reported clinical practice, not measured.*
- Geometry is regular: axial slices, coplanar, equally spaced, isotropic voxels for the Teeth profile (0.07 mm). No gantry tilt. Single-frame uncompressed DICOM. *The "uncompressed" and "single-frame" parts ARE verified by the user (his area of expertise). The "no gantry tilt / coplanar / equispaced" parts should be checked against `ImageOrientationPatient` and `ImagePositionPatient` of the test scan.*

**Implication for registration:** the natural rotation center for any in-plane rigid transform is the **isocenter projected into the slice's pixel grid**, not the geometric center of the image array. Today `register_slice` uses image center — this is wrong by design and is one of the suspects for bug B1 in `05_STATUS.md`. Compute the isocenter pixel as:

```
ipp = ImagePositionPatient        # 3-vector in mm
spacing = PixelSpacing             # (sy, sx) in mm  (DICOM order is row, col)
iso_col = (0.0 - ipp[0]) / spacing[1]
iso_row = (0.0 - ipp[1]) / spacing[0]
```

(z is irrelevant for an in-slice 2D transform.)

## Motion priors (hypothesis from operator intuition)

These are *plausible defaults* to bias the algorithm and the report visualization, not measured statistics. If real-world data contradicts them, update this table — do not force the data to fit.

Probability that real patient motion occurs along each axis during a CBCT scan:

| Axis / DOF | Probability | Cause |
|---|---|---|
| **Y translation (vertical, sup-inf)** | **HIGH** | Breathing. Dominant motion. Quasi-periodic. |
| X translation (lateral) | LOW | Patient does not slide sideways. |
| Z translation (anteroposterior) | LOW | Patient does not slide forward/back. |
| In-plane rotation (around Z) | LOW–MEDIUM | Possible if head turns; head support usually prevents it. |
| Out-of-plane rotation (around X or Y) | RARE | Possible in vestibular disorders / instability. Excluded from MVP. |

## Algorithm consequences

1. **2D rigid slice-to-slice is defensible** for this protocol. Out-of-plane motion is rare; in-plane Y translation (the dominant case) is fully representable by the 2D model. Earlier docs implied 3D was needed — for **this** protocol it is not.

2. **Anisotropic safety limits.** A given displacement in Y is more likely to be real motion; the same displacement in X is more likely to be a registration artifact. The current single `max_translation_mm` field in `TeethProfile` is a simplification. Better:
   ```
   max_translation_y_mm = 1.5    # generous, breathing is real
   max_translation_x_mm = 0.5    # tight, lateral motion is suspicious
   ```
   When refactoring `Profile` for the hexagonal architecture (`DOC/06`), split this field. Until then, the single value is a compromise calibrated for Y.

3. **Sanity check on phase-correlation output.** If `|tx| > |ty|` consistently across the volume, something is wrong — either an axis swap in the code or a real but unusual scenario (operator error, severe head turn). Worth a warning in the report.

4. **Report visualization.** The displacement-per-slice plot must show `tx` and `ty` as **separate traces**, not magnitude. Expected shapes:
   - `ty` curve: low-frequency oscillation, amplitude up to ~10 px, possibly with drift.
   - `tx` curve: flat near zero, occasional spikes are noise.
   - If both curves look similar, the algorithm is broken or the patient was severely uncooperative.

5. **Rotation expectation.** The recovered rotation per slice should be near zero with rare small spikes. A volume where most slices show rotation > 0.5° is suspicious — re-check input geometry assumptions before trusting the output.

## What this does NOT change

- Pipeline structure (phase corr → SimpleITK MI rigid → safety gate → resample) stays the same.
- Sign conventions in `03_ARCHITECTURE_NOW.md` stay the same.
- Hexagonal target in `06_HEXAGONAL_TARGET.md` stays the same — these motion priors live inside the **domain** layer (`Profile`, `MotionLimits`), not in adapters.
