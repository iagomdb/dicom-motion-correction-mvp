from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import pydicom
from pydicom.uid import generate_uid


def _slice_sort_key(ds: pydicom.Dataset) -> float:
    if hasattr(ds, "ImageOrientationPatient") and hasattr(ds, "ImagePositionPatient"):
        iop = np.array(ds.ImageOrientationPatient, dtype=np.float64)
        row, col = iop[:3], iop[3:]
        normal = np.cross(row, col)
        ipp = np.array(ds.ImagePositionPatient, dtype=np.float64)
        return float(np.dot(normal, ipp))
    return float(getattr(ds, "InstanceNumber", 0))


def load_dicom_series(input_dir: str | Path) -> tuple[list[pydicom.Dataset], np.ndarray, dict]:
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Not a directory: {input_dir}")

    candidates: list[tuple[Path, pydicom.Dataset]] = []
    for entry in os.scandir(input_dir):
        if not entry.is_file():
            continue
        try:
            ds = pydicom.dcmread(entry.path, stop_before_pixels=True, force=True)
        except Exception:
            continue
        if not hasattr(ds, "SOPClassUID"):
            continue
        if (0x7FE0, 0x0010) not in ds:
            # Pixel data tag check requires full file; trust SOPClassUID + try full read later.
            pass
        candidates.append((Path(entry.path), ds))

    if not candidates:
        raise ValueError(f"No DICOM files found in {input_dir}")

    candidates.sort(key=lambda t: _slice_sort_key(t[1]))

    datasets: list[pydicom.Dataset] = []
    arrays: list[np.ndarray] = []
    ref_shape: tuple[int, int] | None = None
    orig_dtype: np.dtype | None = None

    for path, _ in candidates:
        full = pydicom.dcmread(str(path), force=True)
        if not hasattr(full, "PixelData"):
            continue
        pix = full.pixel_array
        if ref_shape is None:
            ref_shape = pix.shape
            orig_dtype = pix.dtype
        elif pix.shape != ref_shape:
            raise ValueError(f"Inconsistent slice shape: {pix.shape} vs {ref_shape} ({path.name})")

        slope = float(getattr(full, "RescaleSlope", 1.0))
        intercept = float(getattr(full, "RescaleIntercept", 0.0))
        if slope != 1.0 or intercept != 0.0:
            pix = pix.astype(np.float32) * slope + intercept

        arrays.append(pix)
        datasets.append(full)

    if not datasets:
        raise ValueError(f"No DICOM files with pixel data in {input_dir}")

    if orig_dtype == np.int16 and all(a.dtype == np.int16 for a in arrays):
        volume = np.stack(arrays, axis=0).astype(np.int16)
    else:
        volume = np.stack(arrays, axis=0).astype(np.float32)

    first = datasets[0]
    ps = getattr(first, "PixelSpacing", [0.0, 0.0])
    info = {
        "num_slices": volume.shape[0],
        "rows": volume.shape[1],
        "cols": volume.shape[2],
        "pixel_spacing": (float(ps[0]), float(ps[1])),
        "slice_thickness": float(getattr(first, "SliceThickness", 0.0)),
        "modality": str(getattr(first, "Modality", "")),
        "series_description": str(getattr(first, "SeriesDescription", "")),
        "manufacturer": str(getattr(first, "Manufacturer", "")),
        "patient_id": str(getattr(first, "PatientID", "")),
    }
    return datasets, volume, info


def save_corrected_series(
    original_datasets: list[pydicom.Dataset],
    corrected_volume: np.ndarray,
    output_dir: str | Path,
    correction_metadata: dict,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(original_datasets) != corrected_volume.shape[0]:
        raise ValueError(
            f"Slice count mismatch: {len(original_datasets)} datasets vs {corrected_volume.shape[0]} volume slices"
        )

    new_series_uid = generate_uid()
    metadata_json = json.dumps(correction_metadata)

    for i, src_ds in enumerate(original_datasets):
        ds = copy.deepcopy(src_ds)
        original_dtype = src_ds.pixel_array.dtype
        slice_data = corrected_volume[i].astype(original_dtype)
        ds.PixelData = slice_data.tobytes()

        ds.SOPInstanceUID = generate_uid()
        ds.SeriesInstanceUID = new_series_uid
        if hasattr(ds, "file_meta") and ds.file_meta is not None:
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

        existing_desc = str(getattr(ds, "SeriesDescription", "")).strip()
        ds.SeriesDescription = (existing_desc + " [Motion Corrected]").strip()

        ds.add_new((0x0099, 0x0010), "LO", "DICOM_MC")
        ds.add_new((0x0099, 0x1001), "UT", metadata_json)

        out_path = output_dir / f"slice_{i:04d}.dcm"
        ds.save_as(str(out_path), write_like_original=False)


def validate_saved_series(output_dir: str | Path, expected_count: int) -> bool:
    output_dir = Path(output_dir)
    files = sorted(output_dir.glob("slice_*.dcm"))
    if len(files) != expected_count:
        return False

    sample = random.sample(files, min(5, len(files)))
    uids: set[str] = set()
    ref_shape: tuple[int, ...] | None = None
    for f in sample:
        ds = pydicom.dcmread(str(f), force=True)
        uids.add(str(ds.SOPInstanceUID))
        shape = ds.pixel_array.shape
        if ref_shape is None:
            ref_shape = shape
        elif shape != ref_shape:
            return False

    return len(uids) == len(sample)
