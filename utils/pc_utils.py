import torch

def downsample_points_fixed(x, mask=None, M=2048, strategy="random"):
    """
    x:    [B, N, 3] float
    mask: [B, N] bool or None (True = valid). If None, all points valid.
    M:    target number of points per scene
    strategy: "random" (with replacement to always return M)
    Returns:
      x_ds:   [B, M, 3]
      idx_ds: [B, M] (indices in the original N per-batch)
    """
    B, N, C = x.shape
    device = x.device

    if mask is None:
        mask = torch.ones(B, N, dtype=torch.bool, device=device)

    # Build per-batch index lists of valid points
    # We’ll do multinomial with replacement to *guarantee* M points returned.
    # Use uniform probs over valid points.
    probs = mask.float()
    # Prevent zero-sum rows (in case a scene is all padding)
    row_sums = probs.sum(dim=1, keepdim=True).clamp_min(1.0)
    probs = probs / row_sums

    # Sample exactly M indices per batch (with replacement -> robust)
    idx_ds = torch.multinomial(probs, num_samples=M, replacement=True)  # [B, M]

    # Gather points
    x_ds = x.gather(1, idx_ds.unsqueeze(-1).expand(-1, -1, C))  # [B, M, 3]
    return x_ds, idx_ds
