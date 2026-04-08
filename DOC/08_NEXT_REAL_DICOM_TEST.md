# Next Step: First Real DICOM Test

**Goal:** take the MVP from "works on synthetic phantom" to "produced a corrected DICOM series from a real Teeth 0.07 mm CBCT scan". This is the first reality check.

**Input location:** `./dicom_example/` (relative to project root). User will place a real uncompressed Teeth 0.07 mm CBCT series there.

**Do NOT** start the hexagonal migration. It is cancelled. Work directly on the current flat package.

## Prereqs (verify before touching code)

- `DOC/05_STATUS.md` — B1 is resolved. Registration is working on synthetic (0.87 px mean error). B2 (missing curand/cublas) is irrelevant here — we only use cufft, already installed.
- `DOC/07_PHYSICS_AND_MOTION_MODEL.md` — contains geometry hypotheses that MUST be confirmed against the real DICOM headers before trusting registration output. The checklist at the top of that file is step 1 of this task.

## Phase A — Header inspection (5 min, no code changes)

Before writing anything, run this one-liner and paste the output into a comment on `DOC/05_STATUS.md` under a new section "Real DICOM inspection":

```bash
"D:/dicom_mc/venv/Scripts/python.exe" -c "
import pydicom, os, glob
files = sorted(glob.glob('dicom_example/*'))
print(f'{len(files)} files')
ds = pydicom.dcmread(files[0])
for tag in ['Modality','Manufacturer','SeriesDescription','Rows','Columns',
            'PixelSpacing','SliceThickness','ImageOrientationPatient',
            'ImagePositionPatient','BitsAllocated','BitsStored','HighBit',
            'PixelRepresentation','RescaleSlope','RescaleIntercept',
            'NumberOfFrames','TransferSyntaxUID']:
    v = getattr(ds, tag, '<missing>')
    if tag == 'TransferSyntaxUID':
        v = ds.file_meta.get('TransferSyntaxUID', '<missing>')
    print(f'  {tag}: {v}')
ds_last = pydicom.dcmread(files[-1])
print('first ipp z:', ds.ImagePositionPatient[2] if hasattr(ds,'ImagePositionPatient') else '?')
print('last  ipp z:', ds_last.ImagePositionPatient[2] if hasattr(ds_last,'ImagePositionPatient') else '?')
print('px dtype:', ds.pixel_array.dtype, 'shape:', ds.pixel_array.shape)
print('px min/max:', ds.pixel_array.min(), ds.pixel_array.max())
" 2>&1
```

Then answer these questions by comparing against `DOC/07`:

1. Is `ImageOrientationPatient` == `[1, 0, 0, 0, 1, 0]`? (axial, no tilt)
2. Is `PixelSpacing` isotropic and near 0.07 mm?
3. Is `TransferSyntaxUID` one of the uncompressed ones (`1.2.840.10008.1.2`, `1.2.840.10008.1.2.1`, `1.2.840.10008.1.2.2`)? If it's JPEG2000 or RLE, **stop** and tell the user — current code cannot handle compressed pixel data.
4. Is `NumberOfFrames` absent or 1? If multi-frame, **stop** and tell the user.
5. Is `PixelRepresentation` 1 (signed) or 0 (unsigned)?
6. Is `BitsStored` < `BitsAllocated`? Record both values — `dicom_io.save_corrected_series` currently clips to the full int16 range which is wrong if `BitsStored < 16`; this is a known gap in the save path (not yet fixed, intentional — wait for real data before changing clipping logic).
7. Does central-slice `ImagePositionPatient[0]` and `[1]` sit near the image center (`(Columns-1)/2 * PixelSpacing[1]` etc. from IPP) or is the isocenter meaningfully offset? Compute `iso_col = -ipp[0]/PixelSpacing[1]`, `iso_row = -ipp[1]/PixelSpacing[0]` and report.

**If any answer violates the hypothesis, do NOT proceed to Phase B.** Report findings and stop. The algorithm assumes the hypothesis holds.

## Phase B — Minimal CLI (~80 lines, `main.py`)

Write `dicom_motion_correction/main.py`. No feature creep.

**Signature:**
```
python -m dicom_motion_correction.main --input <dir> --output <dir> [--dry-run] [--cpu]
```

**Required behavior:**
1. `argparse` with only those 4 flags. No profile selection (only `teeth_007` exists), no threshold overrides, no report flag (yet).
2. Validate `--input` exists and has `.dcm` files. Validate `--output` does not exist OR is empty (refuse to overwrite — clinical safety).
3. Call `dicom_io.load_dicom_series(input_dir)` → datasets, volume, info.
4. Print a one-block header:
   ```
   Input:  <dir>
   Series: <N> slices | <rows>x<cols> | voxel: <pixel_spacing> mm | modality: <...>
   Backend: <GPUBackend.summary()>
   ```
5. Extract `pixel_spacing_mm = float(info['pixel_spacing'][0])` (assume isotropic — error if not).
6. Instantiate `GPUBackend(force_cpu=args.cpu)`.
7. Call `correct_volume(volume, profile, backend, pixel_spacing_mm, progress_callback=tqdm_callback)`.
8. Compute summary from the corrections list:
   - Count `was_corrected=True`
   - Count `rejected_reason="exceeds_limits"`
   - Count `rejected_reason="below_threshold"`
   - Count `rejected_reason="reference"`
   - Mean and max |translation| in px AND mm, mean and max |rotation| in deg
   - Mean metric improvement (before - after; MI is minimized so positive is better)
9. Print the summary block.
10. If `--dry-run`: stop here. Otherwise: `save_corrected_series(datasets, corrected, output_dir, correction_metadata)` where `correction_metadata` is a dict with: timestamp, profile name, tool version from `__init__.__version__`, counts from step 8, and the per-slice list of `(slice_index, rot, tx_px, ty_px, was_corrected, rejected_reason)` as JSON-serializable rows.
11. Call `validate_saved_series(output_dir, N)` and print `OK` or the first error.
12. Print the experimental warning required by `DOC/01_PURPOSE.md`:
    ```
    WARNING: This tool is experimental. Results must be validated by a qualified
    professional before any diagnostic use.
    ```

**Hard constraints:**
- Use `pathlib.Path` throughout.
- No logging framework. Plain `print`. This is a CLI.
- No `try/except` around the core pipeline — let it crash with a traceback if registration or I/O fails. The only `try/except` allowed is around `validate_saved_series` at the very end (post-write validation is the one place an error should not mask the successful write).
- `progress_callback` for `correct_volume`: use `tqdm` wrapping `range(N)`. Callback signature is `(current, total)` per `registration.py`. Use `tqdm.update(1)` per call.
- Do NOT modify `dicom_io.save_corrected_series` bit-clipping logic yet. If Phase A revealed `BitsStored < BitsAllocated`, note it in the status doc and leave a `# TODO bit clipping` comment at the call site. Do not fix it in this pass — fixing requires confirming with the user first (see memory note on pixel format authority).

## Phase C — Run it (the moment of truth)

```bash
cd "c:/Users/iago/Desktop/Geral/VS CODE/Dicom"
"D:/dicom_mc/venv/Scripts/python.exe" -m dicom_motion_correction.main --input dicom_example --output D:/dicom_mc/data/teeth_first_run --dry-run
```

Dry-run first. Confirm the summary looks sane:
- Number of corrected slices should be nonzero but less than total (if every slice is "corrected" there is probably a bug).
- Rejected-exceeds-limits count should be 0 or near 0 for a normal scan. A high count means either the scan is garbage or the algorithm is misbehaving.
- Max translation should be within `max_translation_mm` of the profile (1.4 mm). Anything larger means the safety gate fired and is fine.
- Mean metric improvement should be **positive** (MI is minimized by SimpleITK, so `before - after > 0` means the fit got better).

**If any of those look off, STOP and write findings in `DOC/05_STATUS.md`. Do not run without `--dry-run` until the dry-run output is plausible.**

Then drop `--dry-run` and run again to actually write the output series. Open one slice from the output in any DICOM viewer to confirm it loads. Compare visually to the same slice in the input.

## Phase D — Report findings

Append a new section to `DOC/05_STATUS.md`:

```
## First real DICOM run (YYYY-MM-DD)

### Header inspection
<paste Phase A output + answers to the 7 questions>

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

Update the "Working" table: add a row `dicom_io.load_dicom_series / save_corrected_series` with evidence = "produced valid output from real Teeth scan, N slices".

## Out of scope for this task

- `metrics.py` — do not write it. Phase D uses inline computation from the corrections list.
- `report.py` — do not write it. Visual report comes after real data is validated.
- Bit-stored clipping fix — do not touch. Flag only.
- Any change to `registration.py` — B1 is resolved, do not re-litigate the sign conventions.
- Hexagonal refactor — cancelled, ignore `DOC/06`.

## Claim marker

Before starting, add to `DOC/05_STATUS.md` Active work section:
```
- [in progress] main.py + first real DICOM test — <agent tag> — YYYY-MM-DD
```
When done, replace with:
```
- [done] main.py + first real DICOM test — see "First real DICOM run" section below
```
