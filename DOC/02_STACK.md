# Stack

## Hardware target
- GPU: NVIDIA RTX 2060, 6 GB VRAM (or Quadro P2000 5 GB)
- Driver 591.44 (CUDA 13.1 capable, runs CUDA 12.x binaries)
- Windows 10/11

## Python environment
- **Python**: 3.11.9, installed at `C:\Users\iago\AppData\Local\Programs\Python\Python311\` (default user install)
- **Venv**: `D:\dicom_mc\venv\` — always invoke via `D:/dicom_mc/venv/Scripts/python.exe`
- **Pip cache**: `D:\dicom_mc\cache` — pass `--cache-dir D:\dicom_mc\cache` to every pip call
- **Code**: `c:\Users\iago\Desktop\Geral\VS CODE\Dicom\dicom_motion_correction\` (lives on C:, intentionally separate from heavy deps)
- **Test data / outputs**: `D:\dicom_mc\data\`

## Installed packages
| Package | Version | Role |
|---|---|---|
| numpy | 2.4.4 | arrays (forced to 2.x by cupy) |
| scipy | 1.17.1 | ndimage (affine_transform, gaussian_filter, label) |
| pydicom | 3.0.2 | DICOM I/O |
| SimpleITK | 2.5.3 | rigid registration backend (CPU) |
| matplotlib | 3.10.8 | report rendering |
| tqdm | 4.67.3 | progress bars |
| pynvml | 13.0.1 | VRAM queries |
| psutil | 7.2.2 | RAM watchdog |
| cupy-cuda12x | 14.0.1 | GPU arrays + FFT |
| nvidia-cuda-nvrtc-cu12 | 12.9.86 | JIT kernel compile (cupy needs this) |
| nvidia-cuda-runtime-cu12 | 12.9.79 | CUDA runtime |
| nvidia-cufft-cu12 | 11.4.1.4 | FFT (phase correlation) |
| nvidia-nvjitlink-cu12 | 12.9.86 | JIT linker |

## Install gotchas
- `cupy-cuda12x` wheel ships **without** the NVIDIA libs. Each cupy submodule (arange, random, fft, linalg) lazy-loads a different DLL. Install them via `nvidia-*-cu12` pip wheels — no system CUDA Toolkit needed.
- **Missing libs known to be needed beyond what's installed**: `nvidia-curand-cu12` (for `cp.random.*`), `nvidia-cublas-cu12`, `nvidia-cusolver-cu12`, `nvidia-cusparse-cu12`. Install of `cublas` currently fails because its build requires `wheel-stub` and pypi connection to `wheel-stub` times out. Workaround: `--no-build-isolation` or retry with longer timeout. **Status: blocked, see `05_STATUS.md`.**
- pip needs `--timeout 300 --retries 5` minimum. Pypi from this network is slow on first contact.
- `pynvml` emits a FutureWarning recommending `nvidia-ml-py`. Cosmetic, ignore.

## Invocation pattern
Always from project root, as a package:
```
"D:/dicom_mc/venv/Scripts/python.exe" -m dicom_motion_correction.<module>
```
Never `python file.py` directly — relative imports will break.
