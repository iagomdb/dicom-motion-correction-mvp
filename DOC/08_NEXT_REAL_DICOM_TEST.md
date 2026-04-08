# Real DICOM Test Plan (partially executed)

> **Status:** phase A (header inspection) was executed. Phase B (CLI + full-volume run) was started but not completed before the project was paused. See `05_STATUS.md` for the performance-ceiling blocker that prevented completion.

**Goal:** take the MVP from "works on synthetic phantom" to "produced a corrected DICOM series from a real Teeth 0.07 mm CBCT scan". This was the first reality check.

**Input location:** `./dicom_example/` (relative to project root), not committed to the repository (see `.gitignore`).

## Prerequisites

- `DOC/05_STATUS.md` — registration pipeline validated on synthetic phantom after B1 resolution.
- `DOC/07_PHYSICS_AND_MOTION_MODEL.md` — geometry hypotheses that must be confirmed against real DICOM headers before trusting registration output.

## Phase A — Header inspection

Before running the pipeline, inspect a representative DICOM file from the series and confirm the geometric and encoding assumptions the pipeline relies on.

```python
import pydicom, glob
files = sorted(glob.glob('dicom_example/*'))
print(f'{len(files)} files')
ds = pydicom.dcmread(files[0])
for tag in ['Modality', 'Manufacturer', 'SeriesDescription', 'Rows', 'Columns',
            'PixelSpacing', 'SliceThickness', 'ImageOrientationPatient',
            'ImagePositionPatient', 'BitsAllocated', 'BitsStored', 'HighBit',
            'PixelRepresentation', 'RescaleSlope', 'RescaleIntercept',
            'NumberOfFrames']:
    print(f'  {tag}: {getattr(ds, tag, "<missing>")}')
print(f'  TransferSyntaxUID: {ds.file_meta.get("TransferSyntaxUID", "<missing>")}')
ds_last = pydicom.dcmread(files[-1])
print(f'first ipp z: {ds.ImagePositionPatient[2]}')
print(f'last  ipp z: {ds_last.ImagePositionPatient[2]}')
print(f'px dtype: {ds.pixel_array.dtype}, shape: {ds.pixel_array.shape}')
print(f'px min/max: {ds.pixel_array.min()}, {ds.pixel_array.max()}')
```

Validation questions to answer from the output:

1. Is `ImageOrientationPatient` == `[1, 0, 0, 0, 1, 0]`? (axial, no tilt)
2. Is `PixelSpacing` isotropic and near 0.07 mm?
3. Is `TransferSyntaxUID` one of the uncompressed ones (`1.2.840.10008.1.2`, `1.2.840.10008.1.2.1`, `1.2.840.10008.1.2.2`)? Compressed pixel data is not supported by the current save path.
4. Is `NumberOfFrames` absent or 1? Multi-frame DICOM is not supported.
5. Is `PixelRepresentation` 1 (signed) or 0 (unsigned)?
6. Is `BitsStored` < `BitsAllocated`? The current `save_corrected_series` clips to the full int16 range, which is technically incorrect if `BitsStored < 16`. Known gap in the save path.
7. Does the central slice `ImagePositionPatient[0]` and `[1]` sit near the image center? Compute `iso_col = -ipp[0]/PixelSpacing[1]`, `iso_row = -ipp[1]/PixelSpacing[0]` and compare to `Columns/2`, `Rows/2`.

**If any answer violates the working hypothesis in `DOC/07`, the pipeline should not be run on that volume** — the geometric assumptions baked into `register_slice` would no longer hold.

### Results recorded from the actual run

- Volume: 720 slices, 720×720, 0.07 mm isotropic
- Uncompressed, axial (`IOP=[1,0,0,0,1,0]`), single-frame
- PixelRepresentation: signed
- **BitsAllocated=16, BitsStored=12** — matches prior domain expectation, not yet handled in the save path
- Isocenter at (360, 360), essentially coinciding with the image center (359.5, 359.5)

## Phase B — Minimal CLI (`main.py`)

CLI signature:

```
python -m dicom_motion_correction.main --input <dir> --output <dir> [--dry-run] [--cpu]
```

Required behavior:

1. `argparse` with only these four flags.
2. Validate that `--input` exists and contains `.dcm` files. Validate that `--output` does not exist or is empty (refuse to overwrite — clinical safety).
3. Call `dicom_io.load_dicom_series(input_dir)`.
4. Print a header block with the series info and backend summary.
5. Extract `pixel_spacing_mm` from the series info (assume isotropic — error if not).
6. Instantiate `GPUBackend(force_cpu=args.cpu)`.
7. Call `correct_volume(...)` with a `tqdm`-backed progress callback.
8. Compute summary counts from the `corrections` list:
   - `was_corrected=True` count
   - Rejected reasons: `exceeds_limits`, `below_threshold`, `reference`
   - Mean and max translation magnitude in px and mm
   - Mean and max rotation in degrees
   - Mean metric improvement (`before - after`; positive means better fit)
9. Print the summary.
10. If `--dry-run`: stop. Otherwise call `save_corrected_series(datasets, corrected, output_dir, metadata)`, where `metadata` is a JSON-serializable dict containing: timestamp, profile name, tool version, summary counts, per-slice list of `(slice_index, rot, tx_px, ty_px, was_corrected, rejected_reason)`.
11. Call `validate_saved_series(output_dir, N)` and print `OK` or the first error.
12. Print the experimental-use warning required by `DOC/01_PURPOSE.md`.

### Implementation constraints

- Use `pathlib.Path` throughout.
- Plain `print`, no logging framework. This is a CLI.
- Do not wrap the core pipeline in `try`/`except`. Let failures surface with a traceback. The only exception is around `validate_saved_series` at the very end (post-write validation should not mask a successful write).
- `tqdm` wraps `range(N)`; callback signature is `(current, total)`.
- Do not modify `dicom_io.save_corrected_series` bit-clipping behavior as part of this task. If phase A revealed `BitsStored < BitsAllocated`, leave a TODO at the call site and document in `05_STATUS.md`.

## Phase C — Dry-run sanity check

```
python -m dicom_motion_correction.main --input dicom_example --output <out-dir> --dry-run
```

Expected shape of the summary:

- **Corrected slices**: nonzero but significantly less than total. A 100% "corrected" ratio would indicate a bug (no slice should be below the motion threshold).
- **Rejected by `exceeds_limits`**: 0 or near 0 on a healthy scan. A high count indicates either a damaged scan or a misbehaving algorithm.
- **Max translation**: within the profile's `max_translation_mm` (1.4 mm default). Values larger than this are rejected by the safety gate.
- **Mean metric improvement**: positive. SimpleITK minimizes MI, so `before - after > 0` indicates the fit improved.

If any of these checks fail, the dry run should be investigated before dropping `--dry-run`.

## Phase D — Report template

Append to `DOC/05_STATUS.md` once the run completes end-to-end:

```
## First real DICOM run (YYYY-MM-DD)

### Header inspection
<phase A output + answers to the 7 questions>

### Hypothesis status
- Geometry (IOP, isotropic, uncompressed, single-frame): [confirmed | violated with details]
- Isocenter near image center: [yes/no, offset = (dx, dy) px]
- Pixel format (bits stored, signed): [...]

### Pipeline run
- Input: <N> slices, <rows>x<cols>, <voxel> mm
- Backend used: GPU / CPU
- Duration: <seconds>
- Corrected: X/N | Rejected exceeds: Y | Below threshold: Z | Reference: W
- Mean translation: <tx> mm / <ty> mm
- Max translation: <tx> mm / <ty> mm
- Mean rotation: <deg>
- Mean MI improvement: <before - after>
- Validation (validate_saved_series): [OK | error]

### Observations
<any surprises, viewer compatibility, visual quality, suspected issues>
```

## Out of scope for this task

- `metrics.py` — phase D uses inline computation from the corrections list.
- `report.py` — visual report comes after real-data validation is stable.
- Bit-stored clipping fix — flag only, not fix.
- Changes to `registration.py` — B1 is resolved; do not re-litigate the sign conventions.
- Hexagonal refactor — see `06_HEXAGONAL_TARGET.md` (cancelled).
