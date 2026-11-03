"""Feature Squeezing: bit-depth reduction + median filter (eval-time)."""
import numpy as np
import torch

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    from scipy.ndimage import median_filter as _sp_median
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def _median_np(arr_bhwc: np.ndarray, kernel: int) -> np.ndarray:
    """arr: (B,H,W,C) uint8 -> median filtered. Быстрый путь cv2 > scipy > PIL."""
    if kernel < 3:
        return arr_bhwc
    if _HAS_CV2:
        out = np.empty_like(arr_bhwc)
        for b in range(arr_bhwc.shape[0]):
            img = np.ascontiguousarray(arr_bhwc[b])
            if img.shape[2] == 3:
                out[b] = cv2.medianBlur(img, kernel)
            else:
                for c in range(img.shape[2]):
                    out[b, :, :, c] = cv2.medianBlur(
                        np.ascontiguousarray(img[:, :, c]), kernel)
        return out
    if _HAS_SCIPY:
        out = np.empty_like(arr_bhwc)
        for b in range(arr_bhwc.shape[0]):
            for c in range(arr_bhwc.shape[3]):
                out[b, :, :, c] = _sp_median(arr_bhwc[b, :, :, c], size=kernel)
        return out
    # медленный fallback
    from PIL import Image, ImageFilter
    out = np.empty_like(arr_bhwc)
    for b in range(arr_bhwc.shape[0]):
        for c in range(arr_bhwc.shape[3]):
            img = Image.fromarray(arr_bhwc[b, :, :, c])
            out[b, :, :, c] = np.array(img.filter(ImageFilter.MedianFilter(size=kernel)))
    return out


def _backend_name() -> str:
    return "cv2" if _HAS_CV2 else ("scipy" if _HAS_SCIPY else "PIL (slow)")


def feature_squeeze_tensor(
    x: torch.Tensor,
    bit_depth: int = 5,
    median_filter_size: int = 2,
    mean: list[float] = DEFAULT_MEAN,
    std: list[float] = DEFAULT_STD,
) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    device, dtype = x.device, x.dtype

    mean_t = torch.tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=dtype).view(1, -1, 1, 1)

    x_denorm = (x * std_t + mean_t).clamp(0.0, 1.0)
    arr = (x_denorm.detach().cpu().numpy() * 255.0).round().astype(np.uint8)  # B,C,H,W

    # bit-depth reduction: 8 -> bit_depth бит
    levels = 2 ** int(bit_depth)
    step = 255.0 / (levels - 1)
    arr = (np.round(arr / step) * step).clip(0, 255).astype(np.uint8)

    # median filter
    kernel = 2 * int(median_filter_size) + 1
    arr_hwc = np.transpose(arr, (0, 2, 3, 1))  # B,H,W,C
    arr_hwc = _median_np(arr_hwc, kernel)

    out_t = torch.from_numpy(arr_hwc).to(device=device, dtype=dtype) / 255.0
    out_t = out_t.permute(0, 3, 1, 2).contiguous()
    return (out_t - mean_t) / std_t