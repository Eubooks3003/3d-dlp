import time
import random
import numpy as np
import torch
import os

def _rng_state_pack():
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

def _rng_state_load(pkg):
    try:
        if "python_random_state" in pkg: random.setstate(pkg["python_random_state"])
        if "numpy_random_state"  in pkg: np.random.set_state(pkg["numpy_random_state"])
        if "torch_rng_state"     in pkg: torch.set_rng_state(pkg["torch_rng_state"])
        if "torch_cuda_rng_state" in pkg and pkg["torch_cuda_rng_state"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(pkg["torch_cuda_rng_state"])
    except Exception as e:
        print(f"[ckpt] Warning: RNG restore failed: {e}")

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric, extra=None):
    ckpt = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": _rng_state_pack(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "extra": extra or {},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)
    return path

def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    obj = torch.load(path, map_location=map_location)
    # allow both “full ckpt” and “weights-only”
    if isinstance(obj, dict) and "model" in obj:
        model.load_state_dict(obj["model"], strict=False)
        if optimizer is not None and obj.get("optimizer") is not None:
            optimizer.load_state_dict(obj["optimizer"])
        if scheduler is not None and obj.get("scheduler") is not None:
            scheduler.load_state_dict(obj["scheduler"])
        _rng_state_load(obj.get("rng_state", {}))
        return {
            "epoch": obj.get("epoch", 0),
            "best_metric": obj.get("best_metric", float("inf")),
            "extra": obj.get("extra", {}),
            "is_full_ckpt": True,
        }
    else:
        # weights-only file
        model.load_state_dict(obj, strict=False)
        return {"epoch": 0, "best_metric": float("inf"), "extra": {}, "is_full_ckpt": False}
