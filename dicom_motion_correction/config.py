from dataclasses import dataclass, field


VRAM_BUDGET_FRACTION = 0.75
DEFAULT_CACHE_DIR = r"D:\dicom_mc\cache"


@dataclass
class TeethProfile:
    motion_threshold_mm: float = 0.05
    max_translation_mm: float = 1.4
    max_rotation_deg: float = 5.0
    reference_method: str = "mean_central_k"
    reference_k: int = 9
    pyramid_levels: int = 3
    pyramid_shrink: list[int] = field(default_factory=lambda: [4, 2, 1])
    pyramid_sigma: list[float] = field(default_factory=lambda: [2.0, 1.0, 0.0])
    mi_histogram_bins: int = 50
    optimizer_iterations: int = 100
    metric_sampling_percentage: float = 0.5
    roi_mask: str = "auto"


@dataclass
class ProductionTeethProfile(TeethProfile):
    pyramid_shrink: list[int] = field(default_factory=lambda: [8, 4, 2])
    pyramid_sigma: list[float] = field(default_factory=lambda: [4.0, 2.0, 1.0])
    optimizer_iterations: int = 40
    metric_sampling_percentage: float = 0.15


def get_profile(name: str = "teeth_007") -> TeethProfile:
    if name == "teeth_007":
        return TeethProfile()
    if name == "teeth_007_prod":
        return ProductionTeethProfile()
    raise ValueError(f"Unknown profile: {name}")
