# Target: Hexagonal Architecture (cancelled — historical)

> **Status:** this migration plan was drafted but never executed. It is kept as a historical reference. The project remained on the flat package layout described in `03_ARCHITECTURE_NOW.md`.

Motivation for the plan at the time of writing: isolate the registration domain from infrastructure (DICOM I/O, GPU runtime, CLI, reports) so that:

- Domain logic is testable without files, GPU, or SimpleITK installed.
- Backends (CPU vs GPU; SimpleITK vs custom) are swappable.
- I/O formats (DICOM today; NIfTI / PNG stack later) are swappable.

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
│   No numpy in signatures (only inside method bodies   │
│   if unavoidable). No sitk, no cupy, no pydicom.      │
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
├── synthetic/              test data generator
│   ├── phantom.py
│   └── motion.py
├── composition_root.py     wires adapters → ports → use cases
└── cli/
    └── main.py             argparse, calls composition_root
```

## Mapping from flat layout to hexagonal

| Flat file | Hexagonal target |
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

## Domain type sketch

```python
# domain/types.py — pure stdlib + dataclasses, no numpy in public types
from dataclasses import dataclass

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

`Volume` and `Slice` are slightly different — they need to carry pixel data, which is unavoidably `numpy.ndarray`. Decision: allow `numpy.ndarray` as a domain primitive (treat it like `bytes`), but never `cupy.ndarray`, never `sitk.Image`, never `pydicom.Dataset` in domain signatures.

## Port sketches

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

## Architectural constraints (had the migration proceeded)

- Use `typing.Protocol` (PEP 544 structural typing), not `abc.ABC`. Adapters do not need to inherit — matching signatures are enough.
- No `import cupy`, no `import SimpleITK`, no `import pydicom` outside `adapters/`. Enforceable later via `import-linter`.
- Domain never imports from ports, application, or adapters. Ports never import from application or adapters. Application imports domain + ports only. Adapters import domain + ports only.

## Why it was cancelled

See `05_STATUS.md` for the full pause rationale. Short version: the CPU-bound performance ceiling of SimpleITK Mattes MI made further investment in the current pipeline uneconomical, and a GPU-native or deep-learning replacement would be a new project rather than a refactor. Restructuring a pipeline that is itself being replaced had no payoff.
