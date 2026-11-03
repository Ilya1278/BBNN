"""JPEG-компрессия как препроцессинг-защита (eval-time)."""
import io
import numpy as np
import torch
from PIL import Image

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def jpeg_compress_tensor(
    x: torch.Tensor,
    quality: int = 75,
    mean: list[float] = DEFAULT_MEAN,
    std: list[float] = DEFAULT_STD,
) -> torch.Tensor:
    """x: (B,C,H,W) нормализованный. Возвращает нормализованный тензор после JPEG."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    device, dtype = x.device, x.dtype

    mean_t = torch.tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=dtype).view(1, -1, 1, 1)

    x_denorm = (x * std_t + mean_t).clamp(0.0, 1.0)
    arr = (x_denorm.detach().cpu().numpy() * 255.0).round().astype(np.uint8)
    arr = np.transpose(arr, (0, 2, 3, 1))  # B,H,W,C

    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        buf = io.BytesIO()
        Image.fromarray(arr[i]).save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        out[i] = np.array(Image.open(buf).convert("RGB"))

    out_t = torch.from_numpy(out).to(device=device, dtype=dtype) / 255.0
    out_t = out_t.permute(0, 3, 1, 2).contiguous()
    return (out_t - mean_t) / std_t