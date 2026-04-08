# Purpose

## Problem
CBCT scans suffer from inter-slice misalignment caused by patient micro-motion during acquisition. At 0.07 mm voxel size, sub-millimeter motion destroys diagnostic value.

## Solution
2D rigid registration of each slice against a reference built from the central slices of the same volume. Output is a new DICOM series (original is never modified).

## Current scope (phase 1)
- **One protocol only**: Teeth FOV 5×5 cm, voxel 0.07 mm isotropic, ~715×715×720, fits in 6 GB VRAM as float32.
- 2D rigid only (rotation + translation). No 3D, no deformable, no chunking.
- CLI tool. No GUI, no PACS, no batch.

## Out of scope (later phases, do not implement)
- Other FOVs (Arch 1024 slices, Full, Face) — need chunking + adaptive reference
- 3D registration
- Deformable registration
- PACS / DICOMweb
- Web UI

## Clinical safety contract
- Output is **always** a new series with new SOPInstanceUIDs and new SeriesInstanceUID.
- Geometric DICOM tags are preserved bit-exact (see `-` hard rules).
- Corrections that exceed conservative limits (5° rotation, ~1.4 mm translation) are **rejected**, not clamped — the original slice is kept untouched.
- Tool output must carry an "experimental, requires professional validation" marker in the saved DICOM private tag and any report.
