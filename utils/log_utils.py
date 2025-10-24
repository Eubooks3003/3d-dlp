import time
import random
import numpy as np
import torch
import os

import wandb

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


def grad_norm(module):
    sq = 0.0
    has = False
    for p in module.parameters():
        if p.grad is not None:
            sq += p.grad.detach().float().norm()**2
            has = True
    return (sq.sqrt().item() if has else 0.0)

def log_block_grads(model, step):
    groups = {
        "prior": getattr(model, "prior", None),
        "obj_dec": getattr(model, "decoder", None) and getattr(model.decoder, "obj_dec", None),
        "bg_dec": getattr(model, "decoder", None) and getattr(model.decoder, "bg_dec", None),
        "pint": getattr(model, "pint_enc", None),
        "feat_enc": getattr(model, "feat_enc_point", None),
    }
    for name, blk in groups.items():
        if blk is None: continue
        wandb.log({f"gradnorm/{name}": grad_norm(blk)}, step=step)

def log_param_updates(model, step, every=50):
    if step % every: return
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if not hasattr(p, "_prev"):
            p._prev = p.detach().clone()
            continue
        denom = p._prev.norm().clamp_min(1e-12)
        drel  = (p.detach() - p._prev).norm() / denom
        # log a few aggregates to avoid spam:
        if ("obj_dec" in name) or ("prior" in name) or ("bg_dec" in name):
            wandb.log({f"update/{name.split('.')[0]}": drel.item()}, step=step)
        p._prev.copy_(p.detach())

import matplotlib.pyplot as plt

def plot_grad_flow(named_params, step):
    ave_grads, layers = [], []
    for n, p in named_params:
        if p.requires_grad and ("bias" not in n) and (p.grad is not None):
            layers.append(n[:30])
            ave_grads.append(p.grad.detach().abs().mean().item())
    if not ave_grads: return
    plt.figure(figsize=(9,3))
    plt.plot(ave_grads); plt.yscale("log"); plt.title("grad flow (mean abs)")
    plt.xlabel("layers"); plt.tight_layout()
    wandb.log({"debug/grad_flow": wandb.Image(plt)}, step=step)
    plt.close()



def wandb_log_iter_losses(all_losses, iteration):
    """Log per-iteration scalar losses to wandb."""
    flat = {}
    for k, v in all_losses.items():
        if v is None: 
            continue
        try:
            flat[f"loss/{k}"] = float(v.detach().mean().item())
        except Exception:
            pass
    if flat:
        wandb.log(flat, step=iteration)


def topk_indices_from_output(model_output, k):
    """
    Choose top-k particles per batch using (in order of preference):
      obj_on (expected presence) → mu_score → z_score → fallback uniform.
    Returns LongTensor [B, k] of indices.
    """
    # obj_on can be probs/logits depending on your encoder; we accept either.
    obj_on = model_output.get("obj_on", None)      # [B,K,1] or [B,K]
    mu_score = model_output.get("mu_score", None)  # [B,K,1]
    z_score  = model_output.get("z_score", None)   # [B,K,1]
    B, K = None, None

    def _shape2(x):
        if x is None: return None
        return x.squeeze(-1) if x.dim() == 3 and x.size(-1) == 1 else x

    cand = None
    for s in (_shape2(obj_on), _shape2(mu_score), _shape2(z_score)):
        if s is not None:
            cand = s
            break

    if cand is None:
        # fallback: all equal scores
        K = model_output["z"].size(1)
        B = model_output["z"].size(0)
        return torch.arange(K, device=model_output["z"].device).unsqueeze(0).repeat(B, 1)[:, :k]

    B, K = cand.shape
    k = min(k, K)
    vals, idx = torch.topk(cand, k=k, dim=1, largest=True)
    return idx


def render_kp_projections(kp_xyz, topk_idx=None, title="kps", step=None):
    """
    kp_xyz: [B,K,3] in [-1,1].
    If topk_idx=[B,k], highlights top-k per batch.
    Logs a matplotlib figure to wandb (one per batch element).
    """
    import matplotlib.pyplot as plt

    B, K, _ = kp_xyz.shape
    for b in range(B):
        xy = kp_xyz[b, :, [0,1]].detach().cpu().numpy()
        xz = kp_xyz[b, :, [0,2]].detach().cpu().numpy()
        yz = kp_xyz[b, :, [1,2]].detach().cpu().numpy()

        sel = None
        if topk_idx is not None:
            sel = topk_idx[b].detach().cpu().numpy()

        fig, axs = plt.subplots(1, 3, figsize=(9, 3), dpi=120)
        for ax, pts, axlbl in zip(
            axs,
            [xy, xz, yz],
            ["XY", "XZ", "YZ"]
        ):
            ax.scatter(pts[:,0], pts[:,1], s=8, alpha=0.6)
            if sel is not None and sel.size > 0:
                ax.scatter(pts[sel,0], pts[sel,1], s=18, marker="x")
            ax.set_xlim([-1,1]); ax.set_ylim([-1,1])
            ax.set_aspect("equal")
            ax.set_title(axlbl)
            ax.axis("off")
        plt.suptitle(title)
        wandb.log({f"viz/{title}_b{b}": wandb.Image(fig)}, step=step)
        plt.close(fig)
