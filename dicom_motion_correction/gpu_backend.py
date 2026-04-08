from __future__ import annotations

import numpy as np

from .config import VRAM_BUDGET_FRACTION


class GPUBackend:
    def __init__(self, device_id: int = 0, vram_budget_mb: int | None = None, force_cpu: bool = False):
        self.device_id = device_id
        self.available = False
        self.cp = None
        self.device_name = "CPU"
        self.total_vram_mb = 0
        self.vram_budget_mb = 0

        if force_cpu:
            return

        try:
            import cupy as cp
            cp.cuda.Device(device_id).use()
            self.cp = cp
            self.available = True
        except Exception:
            self.cp = None
            self.available = False
            return

        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            raw_name = pynvml.nvmlDeviceGetName(handle)
            self.device_name = raw_name.decode() if isinstance(raw_name, bytes) else raw_name
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.total_vram_mb = int(mem.total // (1024 * 1024))
        except Exception:
            free, total = self.cp.cuda.Device(device_id).mem_info
            self.total_vram_mb = int(total // (1024 * 1024))
            self.device_name = f"CUDA Device {device_id}"

        self.vram_budget_mb = vram_budget_mb if vram_budget_mb is not None else int(self.total_vram_mb * VRAM_BUDGET_FRACTION)

    @property
    def free_vram_mb(self) -> int:
        if not self.available:
            return 0
        free, _ = self.cp.cuda.Device(self.device_id).mem_info
        return int(free // (1024 * 1024))

    @property
    def xp(self):
        return self.cp if self.available else np

    def to_device(self, np_array):
        if not self.available:
            return np_array
        return self.cp.asarray(np_array)

    def to_host(self, arr) -> np.ndarray:
        if not self.available:
            return np.asarray(arr)
        if isinstance(arr, np.ndarray):
            return arr
        return self.cp.asnumpy(arr)

    def free_pool(self) -> None:
        if self.available:
            self.cp.get_default_memory_pool().free_all_blocks()

    def summary(self) -> str:
        if not self.available:
            return "CPU only (numpy)"
        return f"GPU: {self.device_name} ({self.total_vram_mb} MB, {self.free_vram_mb} MB free)"


if __name__ == "__main__":
    backend = GPUBackend()
    print(backend.summary())
