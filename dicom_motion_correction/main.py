"""CLI entry point for DICOM motion correction."""
from __future__ import annotations

import argparse
import datetime as dt
import math
import time
from pathlib import Path

from tqdm import tqdm

from . import __version__
from .config import get_profile
from .dicom_io import load_dicom_series, save_corrected_series, validate_saved_series
from .gpu_backend import GPUBackend
from .registration import correct_volume


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="dicom_motion_correction")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def _validate_paths(input_dir: Path, output_dir: Path) -> None:
    if not input_dir.is_dir():
        raise SystemExit(f"--input is not a directory: {input_dir}")
    has_dcm = any(input_dir.rglob("*.dcm"))
    if not has_dcm:
        raise SystemExit(f"--input contains no .dcm files: {input_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"--output exists and is not empty (refusing to overwrite): {output_dir}")


def main() -> None:
    args = _parse_args()
    input_dir: Path = args.input
    output_dir: Path = args.output
    _validate_paths(input_dir, output_dir)

    datasets, volume, info = load_dicom_series(input_dir)

    ps = info["pixel_spacing"]
    if abs(ps[0] - ps[1]) > 1e-6:
        raise SystemExit(f"PixelSpacing is not isotropic: {ps}")
    pixel_spacing_mm = float(ps[0])

    backend = GPUBackend(force_cpu=args.cpu)
    profile = get_profile()

    print(f"Input:  {input_dir}")
    print(f"Series: {info['num_slices']} slices | {info['rows']}x{info['cols']} | "
          f"voxel: {pixel_spacing_mm} mm | modality: {info['modality']}")
    print(f"Backend: {backend.summary()}")

    # TODO bit clipping: real scan has BitsStored=12 < BitsAllocated=16;
    # save_corrected_series currently clips to full int16 range — revisit.

    N = info["num_slices"]
    pbar = tqdm(total=N, desc="Registering", unit="slice")

    def progress_callback(current: int, total: int) -> None:
        pbar.update(1)

    t0 = time.time()
    corrected, corrections = correct_volume(
        volume, profile, backend, pixel_spacing_mm, progress_callback=progress_callback,
    )
    pbar.close()
    duration_s = time.time() - t0

    n_corrected = sum(1 for c in corrections if c.was_corrected)
    n_exceeds = sum(1 for c in corrections if c.rejected_reason == "exceeds_limits")
    n_below = sum(1 for c in corrections if c.rejected_reason == "below_threshold")
    n_reference = sum(1 for c in corrections if c.rejected_reason == "reference")

    active = [c for c in corrections if c.was_corrected]
    if active:
        tx_abs = [abs(c.translation_x) for c in active]
        ty_abs = [abs(c.translation_y) for c in active]
        rot_abs = [abs(c.rotation_deg) for c in active]
        mean_tx_px = sum(tx_abs) / len(tx_abs)
        mean_ty_px = sum(ty_abs) / len(ty_abs)
        max_tx_px = max(tx_abs)
        max_ty_px = max(ty_abs)
        mean_rot = sum(rot_abs) / len(rot_abs)
        max_rot = max(rot_abs)
    else:
        mean_tx_px = mean_ty_px = max_tx_px = max_ty_px = 0.0
        mean_rot = max_rot = 0.0

    metric_improvements = [
        (c.metric_before - c.metric_after)
        for c in active
        if not (math.isnan(c.metric_before) or math.isnan(c.metric_after))
    ]
    mean_metric_improvement = (
        sum(metric_improvements) / len(metric_improvements) if metric_improvements else float("nan")
    )

    print()
    print(f"Duration: {duration_s:.1f} s")
    print(f"Corrected:         {n_corrected}/{N}")
    print(f"Rejected exceeds:  {n_exceeds}")
    print(f"Below threshold:   {n_below}")
    print(f"Reference slices:  {n_reference}")
    print(f"Mean |tx|: {mean_tx_px:.3f} px / {mean_tx_px * pixel_spacing_mm:.4f} mm    "
          f"Mean |ty|: {mean_ty_px:.3f} px / {mean_ty_px * pixel_spacing_mm:.4f} mm")
    print(f"Max  |tx|: {max_tx_px:.3f} px / {max_tx_px * pixel_spacing_mm:.4f} mm    "
          f"Max  |ty|: {max_ty_px:.3f} px / {max_ty_px * pixel_spacing_mm:.4f} mm")
    print(f"Mean |rot|: {mean_rot:.3f} deg   Max |rot|: {max_rot:.3f} deg")
    print(f"Mean MI improvement (before - after): {mean_metric_improvement:+.5f}")

    if args.dry_run:
        print()
        print("Dry run: not writing output.")
        print()
        print("WARNING: This tool is experimental. Results must be validated by a qualified")
        print("professional before any diagnostic use.")
        return

    correction_metadata = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "tool_version": __version__,
        "profile": "teeth_007",
        "counts": {
            "corrected": n_corrected,
            "rejected_exceeds_limits": n_exceeds,
            "below_threshold": n_below,
            "reference": n_reference,
            "total": N,
        },
        "per_slice": [
            {
                "slice_index": c.slice_index,
                "rotation_deg": c.rotation_deg,
                "translation_x_px": c.translation_x,
                "translation_y_px": c.translation_y,
                "was_corrected": c.was_corrected,
                "rejected_reason": c.rejected_reason,
            }
            for c in corrections
        ],
    }

    # TODO bit clipping: see note above.
    save_corrected_series(datasets, corrected, output_dir, correction_metadata)

    try:
        ok = validate_saved_series(output_dir, N)
        print(f"Validation: {'OK' if ok else 'FAILED'}")
    except Exception as e:  # post-write validation errors must not mask the successful write
        print(f"Validation: ERROR ({e})")

    print()
    print("WARNING: This tool is experimental. Results must be validated by a qualified")
    print("professional before any diagnostic use.")


if __name__ == "__main__":
    main()
