# Current Architecture (pre-hexagonal)

Flat package, no layering. Will be refactored — see `06_HEXAGONAL_TARGET.md`.

## Module map

```
dicom_motion_correction/
├── __init__.py        version only
├── config.py          TeethProfile dataclass + get_profile()
├── gpu_backend.py     GPUBackend class — cupy or numpy fallback
├── dicom_io.py        load_dicom_series, save_corrected_series, validate_saved_series
├── synthetic.py       phantom generator + motion injection + recovery error
└── registration.py    phase_correlation_translation, make_body_mask,
                       register_slice, correct_volume
```

Not yet written: `metrics.py`, `report.py`, `main.py` (CLI).

## Dependency graph

```
main (todo)  →  registration  →  gpu_backend  →  cupy/numpy
                ↓                  ↑
                config             pynvml
                ↑
              (synthetic uses no GPU)
```

`synthetic.py` is test-only and depends on nothing inside the package except `InjectedMotion` types. Safe to import in any test.

## Data flow (intended end-to-end)

```
DICOM dir
  → dicom_io.load_dicom_series → (datasets[], volume[Z,Y,X], info{})
  → registration.correct_volume(volume, profile, backend, pixel_spacing_mm)
       internally:
         build reference = mean of central k slices
         build body mask from reference (Otsu + closing + largest CC)
         for each non-reference slice:
           phase_correlation_translation → coarse (tx, ty)
           SimpleITK Euler2D MI rigid → final (rot, tx, ty)
           clamp/reject if exceeds limits
           clamp/skip if below threshold
           else resample
  → (corrected_volume, corrections[])
  → dicom_io.save_corrected_series(datasets, corrected_volume, out_dir, metadata)
  → metrics + report (todo)
```

## Sign conventions (read before touching math)

Each module that handles geometry has a header docstring stating its convention. Two are in play and they don't match — conversion happens at one specific point in `registration.register_slice`.

| Where | Convention |
|---|---|
| `synthetic.InjectedMotion` | Forward motion applied to clean slice. +tx_px = content shifts right. +rotation_deg = CCW around center. |
| `phase_correlation_translation` return | The shift that aligns `moving` onto `reference` (i.e. inverse of corruption). Returned in pixels. |
| `SliceCorrection` (registration output, public) | Same as InjectedMotion convention but **negated** — describes the motion that was *removed*. So `injected + recovered ≈ 0` is the recovery test. |
| SimpleITK `Euler2DTransform` translation | Physical units. Resample direction (fixed → moving). Internal only. Negated when converting to/from `SliceCorrection`. |

The conversion between phase-correlation pixels and SimpleITK physical units happens in `register_slice` lines ~140-160 and ~195-205. **This is where the current bug lives** (see `05_STATUS.md`).

## Backend abstraction

`GPUBackend.xp` returns `cupy` if available, else `numpy`. Code uses `xp = backend.xp; xp.fft.fft2(...)` and stays backend-agnostic. `to_device` / `to_host` / `free_pool` handle the transitions. `force_cpu=True` in the constructor forces numpy fallback for testing without GPU.
