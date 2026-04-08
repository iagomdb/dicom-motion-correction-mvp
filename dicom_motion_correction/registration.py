"""2D rigid registration for CBCT motion correction.

Pipeline per slice:
  1. GPU phase correlation -> coarse integer+subpixel translation
  2. SimpleITK Mattes MI rigid (Euler2D) refinement with multi-resolution
     pyramid, initialized from the phase correlation estimate.

Sign convention: returned SliceCorrection translation/rotation is the
transform that maps the MOVING slice back onto the REFERENCE (i.e. the
INVERSE of any motion that was applied to the slice).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from .config import TeethProfile, get_profile
from .gpu_backend import GPUBackend


@dataclass
class SliceCorrection:
    slice_index: int
    rotation_deg: float
    translation_x: float
    translation_y: float
    metric_before: float
    metric_after: float
    was_corrected: bool
    rejected_reason: str | None = None


def _parabolic_subpixel(r_line: np.ndarray, peak: int) -> float:
    """3-point parabolic interpolation offset in [-0.5, 0.5]."""
    n = r_line.shape[0]
    if peak <= 0 or peak >= n - 1:
        return 0.0
    y0, y1, y2 = float(r_line[peak - 1]), float(r_line[peak]), float(r_line[peak + 1])
    denom = (y0 - 2.0 * y1 + y2)
    if abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (y0 - y2) / denom


def phase_correlation_translation(
    moving: np.ndarray,
    reference: np.ndarray,
    backend: GPUBackend,
    mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return (tx, ty) such that shifting `moving` by (+tx along cols,
    +ty along rows) aligns it to `reference`. This is the INVERSE of the
    motion that corrupted the slice. For a SimpleITK Euler2D initial
    translation (which maps fixed->moving) pass the negation."""
    xp = backend.xp
    ref = xp.asarray(reference, dtype=xp.float32)
    mov = xp.asarray(moving, dtype=xp.float32)
    if mask is not None:
        m = xp.asarray(mask, dtype=xp.float32)
        ref = ref * m
        mov = mov * m

    F1 = xp.fft.fft2(ref)
    F2 = xp.fft.fft2(mov)
    R = F1 * xp.conj(F2)
    R = R / (xp.abs(R) + 1e-10)
    r = xp.fft.ifft2(R).real

    H, W = r.shape
    flat_idx = int(xp.argmax(r))
    peak_y, peak_x = divmod(flat_idx, W)

    # Sub-pixel parabolic refinement (skipped at borders).
    r_host = backend.to_host(r)
    dy = _parabolic_subpixel(r_host[:, peak_x], peak_y)
    dx = _parabolic_subpixel(r_host[peak_y, :], peak_x)

    # Wraparound: FFT shift lives in [0, N); map to signed shift.
    py = peak_y + dy
    px = peak_x + dx
    if py > H / 2:
        py -= H
    if px > W / 2:
        px -= W
    return float(px), float(py)


def make_body_mask(slice_2d: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    smoothed = ndimage.gaussian_filter(slice_2d.astype(np.float32), sigma=sigma)

    # Inline Otsu
    finite = smoothed[np.isfinite(smoothed)]
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.ones_like(slice_2d, dtype=np.float32)
    hist, edges = np.histogram(finite, bins=256, range=(lo, hi))
    hist = hist.astype(np.float64)
    total = hist.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])
    w_bg = np.cumsum(hist)
    w_fg = total - w_bg
    mu_bg = np.cumsum(hist * centers) / np.maximum(w_bg, 1e-12)
    mu_total = (hist * centers).sum() / max(total, 1e-12)
    mu_fg = (mu_total * total - np.cumsum(hist * centers)) / np.maximum(w_fg, 1e-12)
    sigma_b2 = w_bg * w_fg * (mu_bg - mu_fg) ** 2
    valid = (w_bg > 0) & (w_fg > 0)
    if not valid.any():
        return np.ones_like(slice_2d, dtype=np.float32)
    sigma_b2[~valid] = -np.inf
    t_idx = int(np.argmax(sigma_b2))
    threshold = centers[t_idx]

    binary = smoothed > threshold
    binary = ndimage.binary_closing(binary, iterations=3)

    labels, n = ndimage.label(binary)
    if n == 0:
        return np.ones_like(slice_2d, dtype=np.float32)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest = int(np.argmax(counts))
    return (labels == largest).astype(np.float32)


def register_slice(
    moving: np.ndarray,
    reference_sitk: sitk.Image,
    backend: GPUBackend,
    profile: TeethProfile,
    slice_index: int,
    pixel_spacing_mm: float,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, SliceCorrection]:
    reference_array = sitk.GetArrayFromImage(reference_sitk)
    # NOTE: do NOT pass the body mask to phase_correlation_translation.
    # Multiplying by a hard binary mask introduces a strong step at the
    # mask boundary, which dominates the cross-power spectrum and pulls
    # the peak to (0, 0). The mask is still used by the MI refinement
    # below via SetMetricFixedMask. See DOC/05_STATUS.md B1.
    tx0, ty0 = phase_correlation_translation(moving, reference_array, backend, mask=None)

    # IMPORTANT: run the SimpleITK refinement in *pixel units* (spacing=1.0),
    # not physical millimetres. The registration is purely in-slice and the
    # physical spacing has no effect on the optimum, but with sub-millimetre
    # voxels (0.07 mm) passing physical spacing to
    # SetOptimizerScalesFromPhysicalShift + RegularStepGradientDescent leaves
    # the optimizer with a learning rate of 1 mm ~= 14 px and a minStep of
    # 1e-4 mm ~= 0.0014 px, which badly overshoots and never converges. Doing
    # registration in pixel units makes the default (1.0, 1e-4) behave as
    # 1 px / 1e-4 px — which is what the algorithm is actually tuned for.
    # See DOC/05_STATUS.md B1.
    ref_px = sitk.GetImageFromArray(reference_array)
    ref_px.SetSpacing((1.0, 1.0))
    moving_sitk = sitk.GetImageFromArray(moving.astype(np.float32))
    moving_sitk.SetSpacing((1.0, 1.0))

    # Rotation center: use sitk.CenteredTransformInitializer to eliminate
    # manual center/origin arithmetic as a source of bugs.
    euler_init = sitk.Euler2DTransform()
    euler_init = sitk.CenteredTransformInitializer(
        ref_px,
        moving_sitk,
        euler_init,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    initial_transform = sitk.Euler2DTransform(euler_init)

    # phase_correlation_translation returns (px, py) = the shift (in cols,
    # rows) that maps `moving` BACK onto `reference` — i.e. the INVERSE of
    # the injected motion. SimpleITK Euler2D translation is the resample map
    # (fixed → moving); empirically (see B1 investigation) the SimpleITK
    # translation converges to +injected_shift, so the phase-corr estimate
    # must be NEGATED when used as init.
    initial_transform.SetTranslation((-float(tx0), -float(ty0)))
    initial_transform.SetAngle(0.0)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(profile.mi_histogram_bins)
    reg.SetMetricSamplingStrategy(sitk.ImageRegistrationMethod.REGULAR)
    reg.SetMetricSamplingPercentage(profile.metric_sampling_percentage)
    reg.MetricUseFixedImageGradientFilterOff()
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=profile.optimizer_iterations,
        gradientMagnitudeTolerance=1e-6,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetShrinkFactorsPerLevel(profile.pyramid_shrink)
    reg.SetSmoothingSigmasPerLevel(profile.pyramid_sigma)
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    if mask is not None:
        mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
        mask_sitk.CopyInformation(ref_px)
        reg.SetMetricFixedMask(mask_sitk)
    reg.SetInitialTransform(initial_transform, inPlace=False)

    try:
        metric_before = float(reg.MetricEvaluate(ref_px, moving_sitk))
    except Exception:
        metric_before = float("nan")

    final_transform = reg.Execute(ref_px, moving_sitk)
    metric_after = float(reg.GetMetricValue())

    # inPlace=False returns a CompositeTransform; grab nested Euler2D.
    if isinstance(final_transform, sitk.CompositeTransform):
        euler = sitk.Euler2DTransform(final_transform.GetNthTransform(0))
    else:
        euler = sitk.Euler2DTransform(final_transform)
    angle_rad = euler.GetAngle()
    rot_deg = math.degrees(angle_rad)
    tx_phys, ty_phys = euler.GetTranslation()
    # ref_px has spacing=1.0, so tx_phys/ty_phys are already in pixels.
    # SliceCorrection reports the INVERSE of the injected motion ("motion
    # that was removed"). Empirically, in pixel-unit registration,
    # SimpleITK's Euler2D converges to +injected translation (hence negate)
    # but the angle comes out already negated (i.e. -injected_angle), so
    # the reported rot_deg is kept as-is (not negated a second time).
    tx_px = -float(tx_phys)
    ty_px = -float(ty_phys)
    # rot_deg is already in the "motion removed" convention. No negation.

    max_tr_px = profile.max_translation_mm / pixel_spacing_mm
    if (abs(rot_deg) > profile.max_rotation_deg
            or abs(tx_px) > max_tr_px
            or abs(ty_px) > max_tr_px):
        return moving, SliceCorrection(
            slice_index, rot_deg, tx_px, ty_px, metric_before, metric_after,
            was_corrected=False, rejected_reason="exceeds_limits",
        )

    thr_px = profile.motion_threshold_mm / pixel_spacing_mm
    if abs(tx_px) < thr_px and abs(ty_px) < thr_px and abs(rot_deg) < 0.1:
        return moving, SliceCorrection(
            slice_index, rot_deg, tx_px, ty_px, metric_before, metric_after,
            was_corrected=False, rejected_reason="below_threshold",
        )

    resampled = sitk.Resample(
        moving_sitk, ref_px, final_transform,
        sitk.sitkLinear, 0.0, moving_sitk.GetPixelID(),
    )
    out = sitk.GetArrayFromImage(resampled)
    return out, SliceCorrection(
        slice_index, rot_deg, tx_px, ty_px, metric_before, metric_after,
        was_corrected=True, rejected_reason=None,
    )


def correct_volume(
    volume: np.ndarray,
    profile: TeethProfile,
    backend: GPUBackend,
    pixel_spacing_mm: float,
    progress_callback=None,
) -> tuple[np.ndarray, list[SliceCorrection]]:
    N = volume.shape[0]
    k = profile.reference_k
    mid = N // 2
    lo = mid - k // 2
    hi = mid + k // 2
    ref_arr = volume[lo:hi + 1].mean(axis=0).astype(np.float32)
    reference_sitk = sitk.GetImageFromArray(ref_arr)
    reference_sitk.SetSpacing((float(pixel_spacing_mm), float(pixel_spacing_mm)))

    mask = make_body_mask(ref_arr) if profile.roi_mask == "auto" else None

    out_volume = np.empty_like(volume, dtype=np.float32)
    corrections: list[SliceCorrection] = []

    for i in range(N):
        if lo <= i <= hi:
            out_volume[i] = volume[i].astype(np.float32)
            corrections.append(SliceCorrection(i, 0.0, 0.0, 0.0, 0.0, 0.0,
                                               was_corrected=False,
                                               rejected_reason="reference"))
            if progress_callback:
                progress_callback(i + 1, N)
            continue
        moving_slice = volume[i].astype(np.float32)
        moving_sitk_input_shape = moving_slice.shape
        _ = moving_sitk_input_shape  # (unused; here for clarity)
        corrected, corr = register_slice(
            moving_slice, reference_sitk, backend, profile, i, pixel_spacing_mm, mask,
        )
        out_volume[i] = corrected
        corrections.append(corr)
        if progress_callback:
            progress_callback(i + 1, N)
        if (i + 1) % 32 == 0:
            backend.free_pool()

    if np.issubdtype(volume.dtype, np.integer):
        info = np.iinfo(volume.dtype)
        out_volume = np.clip(out_volume, info.min, info.max).astype(volume.dtype)
    return out_volume, corrections


if __name__ == "__main__":
    from . import synthetic

    # TEMPORARY: force CPU backend so the algorithm smoke test does not depend
    # on the CuPy/NVRTC install being finished in parallel.
    backend = GPUBackend(force_cpu=True)
    print("Backend:", backend.summary())

    profile = get_profile()
    clean = synthetic.make_phantom_volume(shape=(32, 128, 128), seed=42)
    corrupted, motions = synthetic.inject_random_motion(clean, n_affected=5, seed=7)

    # Capture coarse phase-corr estimate per affected slice for sanity print.
    k = profile.reference_k
    mid = corrupted.shape[0] // 2
    ref_arr = corrupted[mid - k // 2: mid + k // 2 + 1].mean(axis=0).astype(np.float32)
    coarse_estimates = {}
    for m in motions:
        tx0, ty0 = phase_correlation_translation(
            corrupted[m.slice_index].astype(np.float32), ref_arr, backend,
        )
        coarse_estimates[m.slice_index] = (tx0, ty0)

    corrected_vol, corrections = correct_volume(
        corrupted, profile, backend, pixel_spacing_mm=0.07,
    )

    err = synthetic.compute_recovery_error(motions, corrections)
    print("Recovery errors:")
    for k_, v_ in err.items():
        print(f"  {k_}: {v_}")

    print("Per-slice coarse (phase-corr) vs refined (final):")
    rec_by_idx = {c.slice_index: c for c in corrections}
    for m in motions:
        c = rec_by_idx.get(m.slice_index)
        cx, cy = coarse_estimates[m.slice_index]
        if c is None:
            continue
        print(f"  z={m.slice_index:3d}  injected=({m.translation_x_px:+.2f},"
              f"{m.translation_y_px:+.2f},{m.rotation_deg:+.2f}deg)  "
              f"coarse=({cx:+.2f},{cy:+.2f})  "
              f"final=({c.translation_x:+.2f},{c.translation_y:+.2f},"
              f"{c.rotation_deg:+.2f}deg)  corrected={c.was_corrected}")
