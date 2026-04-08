# Target: Hexagonal Architecture

Migration from the current flat package. Goal: isolate the registration **domain** from infrastructure (DICOM I/O, GPU runtime, CLI, reports) so that:

- Domain logic is testable without files, GPU, or SimpleITK installed.
- Backends (CPU vs GPU; SimpleITK vs ITK-GPU vs custom) are swappable.
- I/O formats (DICOM today; NIfTI/PNG-stack tomorrow) are swappable.
- Multiple Claude agents can work in parallel on adapters without touching the core.

## Layers

```
┌──────────────────────────────────────────────────────┐
│                  ADAPTERS (driving)                   │
│   cli/    api/    notebook/                           │
└────────────────────────┬─────────────────────────────┘
                         │ uses ports
┌────────────────────────▼─────────────────────────────┐
│                    APPLICATION                        │
│   use cases: CorrectVolumeUseCase, ValidateUseCase    │
│   orchestrates domain + ports, no algorithm logic     │
└────────────────────────┬─────────────────────────────┘
                         │ depends on
┌────────────────────────▼─────────────────────────────┐
│                      DOMAIN                           │
│   pure: Volume, Slice, Reference, Correction,        │
│   MotionLimits, Profile                              │
│   policies: rejection rules, threshold checks         │
│   NO numpy in signatures (only inside method bodies   │
│   if unavoidable). NO sitk, NO cupy, NO pydicom.      │
└────────────────────────▲─────────────────────────────┘
                         │ implements
┌────────────────────────┴─────────────────────────────┐
│                  PORTS (interfaces)                   │
│   VolumeReaderPort, VolumeWriterPort,                 │
│   ReferenceBuilderPort, RegistrarPort, MaskerPort,    │
│   ResamplerPort, ComputeBackendPort, ReportPort       │
└────────────────────────▲─────────────────────────────┘
                         │ implements
┌────────────────────────┴─────────────────────────────┐
│                 ADAPTERS (driven)                     │
│   io_dicom/      → VolumeReader/Writer (pydicom)      │
│   compute_cupy/  → ComputeBackend (cupy + nvidia)     │
│   compute_numpy/ → ComputeBackend (cpu fallback)      │
│   registrar_sitk/→ Registrar (SimpleITK Mattes MI)    │
│   reference_mean/→ ReferenceBuilder (mean of central) │
│   masker_otsu/   → Masker (Otsu + closing + cc)       │
│   report_mpl/    → Report (matplotlib PNG)            │
└──────────────────────────────────────────────────────┘
```

## Target folder layout

```
dicom_motion_correction/
├── domain/
│   ├── __init__.py
│   ├── types.py            Volume, Slice, Reference, Correction, Profile, MotionLimits
│   ├── policies.py         pure functions: should_reject, is_below_threshold
│   └── errors.py           domain exceptions (no infra leak)
├── ports/
│   ├── __init__.py
│   ├── reader.py           VolumeReaderPort (Protocol)
│   ├── writer.py           VolumeWriterPort
│   ├── compute.py          ComputeBackendPort
│   ├── registrar.py        RegistrarPort, ReferenceBuilderPort, MaskerPort, ResamplerPort
│   └── report.py           ReportPort
├── application/
│   ├── __init__.py
│   ├── correct_volume.py   CorrectVolumeUseCase
│   └── validate_synthetic.py ValidateSyntheticUseCase
├── adapters/
│   ├── io_dicom/           pydicom impl of reader/writer
│   ├── compute_cupy/       cupy impl of ComputeBackend
│   ├── compute_numpy/      numpy fallback
│   ├── registrar_sitk/     SimpleITK Mattes MI rigid impl
│   ├── reference_mean/     mean-of-k-central impl
│   ├── masker_otsu/        Otsu body mask impl
│   └── report_mpl/         matplotlib report impl
├── synthetic/              test data (not part of runtime, kept here for now)
│   ├── phantom.py
│   └── motion.py
├── composition_root.py     wires adapters → ports → use cases
└── cli/
    └── main.py             argparse, calls composition_root
```

## Mapping from current code

| Current file | Goes to |
|---|---|
| `config.py::TeethProfile` | `domain/types.py::Profile` (frozen dataclass) |
| `config.py::get_profile` | `composition_root.py` (config selection is wiring, not domain) |
| `gpu_backend.py::GPUBackend` | split: `ports/compute.py::ComputeBackendPort` + `adapters/compute_cupy/`, `adapters/compute_numpy/` |
| `dicom_io.py::load_dicom_series` | `adapters/io_dicom/reader.py` implementing `VolumeReaderPort` |
| `dicom_io.py::save_corrected_series` | `adapters/io_dicom/writer.py` implementing `VolumeWriterPort` |
| `registration.py::phase_correlation_translation` | `adapters/registrar_sitk/phase_corr.py` (uses ComputeBackendPort, not cupy directly) |
| `registration.py::make_body_mask` | `adapters/masker_otsu/otsu.py` implementing `MaskerPort` |
| `registration.py::register_slice` | split: `adapters/registrar_sitk/sitk_rigid.py` (Registrar impl) + `domain/policies.py` (rejection rules) |
| `registration.py::correct_volume` | `application/correct_volume.py::CorrectVolumeUseCase` (orchestration only, no math) |
| `synthetic.py` | `synthetic/phantom.py` + `synthetic/motion.py` |

## Domain types (sketch)

```python
# domain/types.py — pure stdlib + dataclasses, no numpy in public types
from dataclasses import dataclass, field

@dataclass(frozen=True)
class MotionLimits:
    motion_threshold_mm: float
    max_translation_mm: float
    max_rotation_deg: float

@dataclass(frozen=True)
class Profile:
    name: str
    limits: MotionLimits
    reference_method: str
    reference_k: int
    pyramid_shrink: tuple[int, ...]
    pyramid_sigma: tuple[float, ...]
    mi_histogram_bins: int
    optimizer_iterations: int
    use_body_mask: bool

@dataclass(frozen=True)
class Correction:
    slice_index: int
    rotation_deg: float
    translation_x_px: float
    translation_y_px: float
    metric_before: float
    metric_after: float
    was_corrected: bool
    rejected_reason: str | None
```

`Volume` and `Slice` are slightly different — they need to carry pixel data, which is unavoidably numpy. Decision: allow `numpy.ndarray` as a domain primitive (treat it like `bytes`), but never `cupy.ndarray`, never `sitk.Image`, never `pydicom.Dataset` in domain signatures.

## Port examples

```python
# ports/registrar.py
from typing import Protocol
from ..domain.types import Profile, Correction
import numpy as np

class RegistrarPort(Protocol):
    def register(
        self,
        moving: np.ndarray,        # 2D
        reference: np.ndarray,     # 2D
        mask: np.ndarray | None,   # 2D, uint8
        pixel_spacing_mm: float,
        profile: Profile,
        slice_index: int,
    ) -> tuple[np.ndarray, Correction]: ...
```

```python
# ports/compute.py
from typing import Protocol, Any

class ComputeBackendPort(Protocol):
    @property
    def name(self) -> str: ...           # "cupy" | "numpy"
    @property
    def available(self) -> bool: ...
    def fft2(self, arr: Any) -> Any: ...
    def ifft2(self, arr: Any) -> Any: ...
    def to_device(self, arr) -> Any: ...
    def to_host(self, arr) -> "np.ndarray": ...
    def free(self) -> None: ...
```

Keeps phase correlation portable — its body uses only `ComputeBackendPort`, no `import cupy`.

## Migration order (suggested for parallel agents)

Each step is independent of the next as long as the domain types land first. Multiple agents can claim non-overlapping bullets.

1. **Foundation (1 agent, blocking):** create `domain/types.py`, `domain/policies.py`, `domain/errors.py`, `ports/*.py`. No behavior — just types and Protocols. Until this lands, nothing else can import from domain/ports.
2. **Synthetic move (1 agent):** copy `synthetic.py` into `synthetic/phantom.py` + `synthetic/motion.py`. No logic changes. Old file stays until step 7.
3. **Compute adapters (1 agent):** port `gpu_backend.py` into `adapters/compute_cupy/` and `adapters/compute_numpy/` implementing `ComputeBackendPort`. New name, narrower API.
4. **DICOM I/O adapters (1 agent):** move `dicom_io.py` into `adapters/io_dicom/`, split read/write, implement ports. Add a fake adapter `adapters/io_memory/` for tests.
5. **Registrar adapter (1 agent, depends on step 1 + 3):** port `register_slice` into `adapters/registrar_sitk/`. **This is also where B1 from `05_STATUS.md` should be fixed** — the cleanup of moving the boundary code into one focused file is the right time to debug the sign convention.
6. **Masker + Reference adapters (1 agent):** trivial extracts.
7. **Application + composition root (1 agent, depends on all above):** write `CorrectVolumeUseCase`, `composition_root.py`, `cli/main.py`. Delete the old flat files (`config.py`, `gpu_backend.py`, `dicom_io.py`, `registration.py`) only after the new pipeline produces the same recovery error on the synthetic phantom.
8. **Validation (1 agent):** add a real test (`tests/test_registration_recovery.py`) that runs `ValidateSyntheticUseCase` and asserts mean error < 0.1 px / 0.1°.

## Rules during migration

- Old files keep working until step 7. No half-broken state on main.
- Each step ends with `python -m dicom_motion_correction.synthetic` still producing the demo PNG (regression smoke test).
- Use `Protocol` (PEP 544 structural typing), not `ABC`. Adapters don't need to inherit, they just need matching signatures.
- No `import cupy`, no `import SimpleITK`, no `import pydicom` outside `adapters/`. Enforced by convention; can be enforced by `import-linter` later.
