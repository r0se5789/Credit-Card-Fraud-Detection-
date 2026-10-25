"""GPU and Device Acceleration Utilities."""

import os
import torch
from typing import Dict, Any, Tuple
from src.utils.logger import logger

_CACHED_DEVICE = None

def get_usable_device() -> Tuple[torch.device, str]:
    """Tests whether CUDA is actually executable for PyTorch tensor operations.
    Handles cutting-edge architectures (e.g. Blackwell sm_120) gracefully.
    """
    global _CACHED_DEVICE
    if _CACHED_DEVICE is not None:
        return _CACHED_DEVICE

    if not torch.cuda.is_available():
        _CACHED_DEVICE = (torch.device("cpu"), "CPU (CUDA not available)")
        return _CACHED_DEVICE

    try:
        # Test tensor kernel execution on device
        t = torch.zeros(2, device="cuda")
        _CACHED_DEVICE = (torch.device("cuda"), f"GPU: {torch.cuda.get_device_name(0)}")
        return _CACHED_DEVICE
    except Exception as e:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "NVIDIA GPU"
        logger.warning(
            f"ℹ️ Hardware contains {gpu_name}, but current PyTorch build does not include sm_120 binary kernels.\n"
            f"   -> PyTorch neural networks will run on optimized multi-threaded CPU.\n"
            f"   -> Tree models (XGBoost) will run directly on the NVIDIA GPU via CUDA."
        )
        _CACHED_DEVICE = (torch.device("cpu"), f"CPU (PyTorch) / GPU for XGBoost ({gpu_name})")
        return _CACHED_DEVICE

def check_gpu_status() -> Dict[str, Any]:
    """Inspects available hardware and prints detailed GPU statistics."""
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    device_name = torch.cuda.get_device_name(0) if cuda_available and device_count > 0 else "None"
    
    usable_device, reason = get_usable_device()

    info = {
        "cuda_available": cuda_available,
        "device_count": device_count,
        "device_name": device_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "usable_device": str(usable_device),
        "status_summary": reason,
    }

    if cuda_available and device_count > 0:
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        info.update({
            "total_vram_gb": round(total_memory_gb, 2),
        })
        logger.info(
            f"🚀 [Hardware Detected] {device_name} | VRAM: {total_memory_gb:.2f} GB | "
            f"CUDA Version: {info['cuda_version']}"
        )
    else:
        info.update({"total_vram_gb": 0.0})
        logger.info("ℹ️ Running on CPU architecture.")

    return info

def clear_gpu_cache():
    """Flushes CUDA memory cache if available."""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
