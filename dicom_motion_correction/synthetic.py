"""Synthetic CBCT phantom + injected motion for ground-truth validation.

Sign convention:
  - InjectedMotion describes the motion artifact APPLIED to a clean slice:
    positive translation_x_px shifts content to the right (toward +x / larger
    column index), positive translation_y_px shifts content downward (toward
    larger row index), positive rotation_deg is counter-clockwise around the
    slice center (standard math convention, as implemented via
    scipy.ndimage.affine_transform with the inverse matrix below).
  - The registration engine returns SliceCorrection describing the transform
    that maps the corrupted `moving` back to the reference. It is therefore
    the INVERSE of the injected motion; recovery error compares
    (injected + recovered) which should be ~0.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage


@dataclass
class InjectedMotion:
    slice_index: int
    rotation_deg: float
    translation_x_px: float
    translation_y_px: float


def make_phantom_volume(
    shape: tuple[int, int, int] = (256, 256, 128),
    dtype=np.float32,
    seed: int = 42,
) -> np.ndarray:
    """Build a CBCT-cross-section-like phantom. shape = (Z, Y, X)."""
    rng = np.random.default_rng(seed)
    Z, Y, X = shape
    vol = np.zeros((Z, Y, X), dtype=np.float32)

    yy, xx = np.mgrid[0:Y, 0:X].astype(np.float32)
    cy, cx = Y / 2.0, X / 2.0

    # Head ellipse: radius varies slowly along Z
    base_ry = 0.42 * Y
    base_rx = 0.42 * X
    for z in range(Z):
        t = (z - Z / 2.0) / Z
        ry = base_ry * (1.0 - 0.08 * t * t)
        rx = base_rx * (1.0 - 0.08 * t * t)
        ellipse = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
        vol[z][ellipse] = 1000.0

    # 6-8 tooth root cylinders on an arc
    n_teeth = 7
    arc_radius = 0.25 * min(Y, X)
    arc_center_y = cy + 0.05 * Y
    tooth_params = []
    for i in range(n_teeth):
        angle = np.pi * (0.15 + 0.7 * i / (n_teeth - 1))  # lower arc
        ty_ = arc_center_y - arc_radius * np.sin(angle)
        tx_ = cx - arc_radius * np.cos(angle)
        radius = 3.0 + 1.5 * rng.random()
        z_lo = int(Z * (0.15 + 0.1 * rng.random()))
        z_hi = int(Z * (0.75 + 0.15 * rng.random()))
        tooth_params.append((ty_, tx_, radius, z_lo, z_hi))

    for (ty_, tx_, radius, z_lo, z_hi) in tooth_params:
        disk = (yy - ty_) ** 2 + (xx - tx_) ** 2 <= radius ** 2
        for z in range(z_lo, min(z_hi, Z)):
            vol[z][disk] = 2000.0

    # High-contrast small features for optimizer corners
    for _ in range(5):
        fy = cy + (rng.random() - 0.5) * 0.5 * Y
        fx = cx + (rng.random() - 0.5) * 0.5 * X
        fr = 1.5 + rng.random() * 1.0
        fz_lo = int(rng.random() * (Z - 10))
        fz_hi = fz_lo + 6
        disk = (yy - fy) ** 2 + (xx - fx) ** 2 <= fr ** 2
        for z in range(fz_lo, fz_hi):
            vol[z][disk] = 3000.0

    # Noise
    vol += rng.normal(0.0, 30.0, size=vol.shape).astype(np.float32)

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        vol = np.clip(vol, info.min, info.max)
    return vol.astype(dtype)


def _affine_matrix_for_slice(rotation_deg: float, tx: float, ty: float,
                             shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return (matrix, offset) for scipy.ndimage.affine_transform so that the
    OUTPUT is the input rotated+translated by (rotation_deg, tx, ty) with
    positive tx shifting right and positive ty shifting down, rotation around
    image center, counter-clockwise positive."""
    H, W = shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    theta = np.deg2rad(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # Forward map (output <- input): p_out = R (p_in - c) + c + t
    # affine_transform computes: p_in = M p_out + offset
    # So M = R^-1, offset = c - R^-1 (c + t)
    R_inv = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float64)
    c = np.array([cy, cx], dtype=np.float64)
    t = np.array([ty, tx], dtype=np.float64)
    offset = c - R_inv @ (c + t)
    return R_inv, offset


def inject_motion(volume: np.ndarray, motions: list[InjectedMotion],
                  order: int = 3) -> np.ndarray:
    """Apply each InjectedMotion to its slice. Positive translation_x_px
    shifts content to the right (+column direction)."""
    out = volume.astype(np.float32, copy=True)
    H, W = volume.shape[1], volume.shape[2]
    for m in motions:
        M, offset = _affine_matrix_for_slice(
            m.rotation_deg, m.translation_x_px, m.translation_y_px, (H, W)
        )
        out[m.slice_index] = ndimage.affine_transform(
            volume[m.slice_index].astype(np.float32),
            matrix=M, offset=offset, order=order, mode="constant", cval=0.0,
        )
    return out


def inject_random_motion(
    volume: np.ndarray,
    n_affected: int,
    seed: int,
    max_rotation_deg: float = 3.0,
    max_translation_px: float = 8.0,
) -> tuple[np.ndarray, list[InjectedMotion]]:
    rng = np.random.default_rng(seed)
    Z = volume.shape[0]
    mid = Z // 2
    reserved = set(range(mid - 4, mid + 5))  # central 9 slices = reference
    candidates = [i for i in range(Z) if i not in reserved]
    if n_affected > len(candidates):
        n_affected = len(candidates)
    chosen = rng.choice(candidates, size=n_affected, replace=False)
    motions: list[InjectedMotion] = []
    for idx in sorted(chosen.tolist()):
        motions.append(InjectedMotion(
            slice_index=int(idx),
            rotation_deg=float(rng.uniform(-max_rotation_deg, max_rotation_deg)),
            translation_x_px=float(rng.uniform(-max_translation_px, max_translation_px)),
            translation_y_px=float(rng.uniform(-max_translation_px, max_translation_px)),
        ))
    corrupted = inject_motion(volume, motions)
    return corrupted, motions


def compute_recovery_error(injected: list[InjectedMotion], recovered: list) -> dict:
    """Match by slice_index. Recovered is the INVERSE of injected, so
    residual = injected + recovered (should be ~0)."""
    rec_by_idx = {r.slice_index: r for r in recovered}
    tx_errs, ty_errs, rot_errs = [], [], []
    for inj in injected:
        if inj.slice_index not in rec_by_idx:
            continue
        r = rec_by_idx[inj.slice_index]
        tx_errs.append(abs(inj.translation_x_px + r.translation_x))
        ty_errs.append(abs(inj.translation_y_px + r.translation_y))
        rot_errs.append(abs(inj.rotation_deg + r.rotation_deg))
    if not tx_errs:
        return {"mean_translation_error_px": float("nan"),
                "max_translation_error_px": float("nan"),
                "mean_rotation_error_deg": float("nan"),
                "max_rotation_error_deg": float("nan"),
                "n_compared": 0}
    tr = [max(a, b) for a, b in zip(tx_errs, ty_errs)]
    return {
        "mean_translation_error_px": float(np.mean(tr)),
        "max_translation_error_px": float(np.max(tr)),
        "mean_rotation_error_deg": float(np.mean(rot_errs)),
        "max_rotation_error_deg": float(np.max(rot_errs)),
        "n_compared": len(tx_errs),
    }


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clean = make_phantom_volume(shape=(32, 128, 128), seed=42)
    corrupted, motions = inject_random_motion(clean, n_affected=5, seed=7)
    print("Injected motions:")
    for m in motions:
        print(f"  z={m.slice_index:3d}  rot={m.rotation_deg:+.3f} deg  "
              f"tx={m.translation_x_px:+.3f} px  ty={m.translation_y_px:+.3f} px")

    affected_idx = motions[len(motions) // 2].slice_index
    clean_s = clean[affected_idx]
    corr_s = corrupted[affected_idx]
    diff = np.abs(clean_s - corr_s)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(clean_s, cmap="gray"); axes[0].set_title(f"Clean z={affected_idx}")
    axes[1].imshow(corr_s, cmap="gray"); axes[1].set_title("Corrupted")
    axes[2].imshow(diff, cmap="hot", vmin=0, vmax=float(diff.max()) * 0.5 + 1e-6)
    axes[2].set_title("|Clean - Corrupted|")
    for a in axes:
        a.axis("off")
    out_path = Path(r"D:/dicom_mc/cache/synthetic_demo.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"Saved {out_path}")
