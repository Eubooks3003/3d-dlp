#!/usr/bin/env python3
"""
export_for_render.py

Export voxel reconstructions from a trained model checkpoint for Blender rendering.

Usage:
    python render/export_for_render.py --log_dir /path/to/logs/run_name [OPTIONS]

The script reads:
    - {log_dir}/hparams.json     -> model config and data path
    - {log_dir}/saves/best.pt    -> model checkpoint

And exports to:
    - {log_dir}/paper_meshes/    -> PLY files for Blender

Examples:
    # Export 10 validation examples
    python render/export_for_render.py --log_dir logs/240126_044143_mimicgen_rgb_coffee

    # Export 20 examples with custom output dir
    python render/export_for_render.py --log_dir logs/my_run --num_examples 20 --output_dir ./my_renders

    # Export specific indices
    python render/export_for_render.py --log_dir logs/my_run --indices 0 5 10 15

    # Use last.pt instead of best.pt
    python render/export_for_render.py --log_dir logs/my_run --ckpt last.pt
"""

import os
import sys
import json
import argparse
import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voxel_models import DLP
from datasets.point_cloud_datasets.get_dataset import get_point_cloud_dataset
from render.export_vox_to_mesh import export_example_for_blender
from utils.log_utils import load_checkpoint


def load_config(log_dir):
    """Load hparams.json from log directory."""
    hparams_path = os.path.join(log_dir, "hparams.json")
    if not os.path.exists(hparams_path):
        raise FileNotFoundError(f"Config not found: {hparams_path}")

    with open(hparams_path, 'r') as f:
        config = json.load(f)

    print(f"[config] Loaded: {hparams_path}")
    return config


def load_model(config, checkpoint_path, device):
    """Initialize and load model from checkpoint.

    Uses the exact same model initialization as train_dlp_voxel.py to ensure compatibility.
    """
    # Extract model parameters from config (same as train_dlp_voxel.py)
    ch = config['ch']
    voxel_grid_whd = config['voxel_grid_whd']
    normalize_rgb = config['normalize_rgb']
    n_kp_per_patch = config['n_kp_per_patch']
    patch_size = config['patch_size']
    anchor_s = config['anchor_s']
    n_kp_enc = config['n_kp_enc']
    n_kp_prior = config['n_kp_prior']
    pad_mode = config.get('pad_mode', 'zeros')
    dropout = config.get('dropout', 0.0)
    features_dist = config.get('features_dist', 'gauss')
    learned_feature_dim = config['learned_feature_dim']
    learned_bg_feature_dim = config.get('learned_bg_feature_dim', learned_feature_dim)
    n_fg_categories = config.get('n_fg_categories', 8)
    n_fg_classes = config.get('n_fg_classes', 4)
    n_bg_categories = config.get('n_bg_categories', 4)
    n_bg_classes = config.get('n_bg_classes', 4)
    scale_std = config.get('scale_std', 0.3)
    offset_std = config.get('offset_std', 0.1)
    obj_on_alpha = config.get('obj_on_alpha', 0.1)
    obj_on_beta = config.get('obj_on_beta', 0.1)
    obj_res_from_fc = config.get('obj_res_from_fc', 4)
    obj_ch_mult = config.get('obj_ch_mult', [1, 2, 4])
    obj_ch_mult_prior = config.get('obj_ch_mult_prior', obj_ch_mult)
    obj_base_ch = config.get('obj_base_ch', 32)
    obj_final_cnn_ch = config.get('obj_final_cnn_ch', 32)
    bg_res_from_fc = config.get('bg_res_from_fc', 8)
    bg_ch_mult = config.get('bg_ch_mult', [1, 2, 4])
    bg_base_ch = config.get('bg_base_ch', 32)
    bg_final_cnn_ch = config.get('bg_final_cnn_ch', 32)
    use_resblock = config.get('use_resblock', False)
    num_res_blocks = config.get('num_res_blocks', 1)
    cnn_mid_blocks = config.get('cnn_mid_blocks', False)
    mlp_hidden_dim = config.get('mlp_hidden_dim', 256)
    pint_enc_layers = config.get('pint_enc_layers', 1)
    pint_enc_heads = config.get('pint_enc_heads', 1)
    separate_depth_features = config.get('separate_depth_features', False)
    depth_feature_dim = config.get('depth_feature_dim', 1)
    split_loss = config.get('split_loss', False)
    depth_loss_ratio = config.get('depth_loss_ratio', 0.25)
    prior_mode = config.get('prior_mode', 'kmeans')
    raw_heatmap_mode = config.get('raw_heatmap_mode', 'luma')

    # Create model with exact same params as train_dlp_voxel.py
    model = DLP(
        cdim=ch,
        image_size=voxel_grid_whd[0],
        normalize_rgb=normalize_rgb,
        n_kp_per_patch=n_kp_per_patch,
        patch_size=patch_size,
        anchor_s=anchor_s,
        n_kp_enc=n_kp_enc,
        n_kp_prior=n_kp_prior,
        pad_mode=pad_mode,
        dropout=dropout,
        features_dist=features_dist,
        learned_feature_dim=learned_feature_dim,
        learned_bg_feature_dim=learned_bg_feature_dim,
        n_fg_categories=n_fg_categories,
        n_fg_classes=n_fg_classes,
        n_bg_categories=n_bg_categories,
        n_bg_classes=n_bg_classes,
        scale_std=scale_std,
        offset_std=offset_std,
        obj_on_alpha=obj_on_alpha,
        obj_on_beta=obj_on_beta,
        obj_res_from_fc=obj_res_from_fc,
        obj_ch_mult_prior=obj_ch_mult_prior,
        obj_ch_mult=obj_ch_mult,
        obj_base_ch=obj_base_ch,
        obj_final_cnn_ch=obj_final_cnn_ch,
        bg_res_from_fc=bg_res_from_fc,
        bg_ch_mult=bg_ch_mult,
        bg_base_ch=bg_base_ch,
        bg_final_cnn_ch=bg_final_cnn_ch,
        use_resblock=use_resblock,
        num_res_blocks=num_res_blocks,
        cnn_mid_blocks=cnn_mid_blocks,
        mlp_hidden_dim=mlp_hidden_dim,
        pint_enc_layers=pint_enc_layers,
        pint_enc_heads=pint_enc_heads,
        timestep_horizon=1,  # Hardcoded to 1, same as train_dlp_voxel.py
        separate_depth_features=separate_depth_features,
        depth_feature_dim=depth_feature_dim,
        split_loss=split_loss,
        depth_loss_ratio=depth_loss_ratio,
        prior_mode=prior_mode,
        raw_heatmap_mode=raw_heatmap_mode,
    )

    # Move model to device
    model = model.to(device)

    # Load checkpoint using the same function as training script
    resume_info = load_checkpoint(checkpoint_path, model, None, None, map_location=device)
    model.eval()

    epoch = resume_info.get('epoch', 'unknown')
    print(f"[model] Loaded checkpoint from epoch {epoch}: {checkpoint_path}")

    return model


def load_dataset(config, mode='val'):
    """Load dataset from config."""
    ds = config['ds']
    root = config['root']
    voxel_grid_whd = config['voxel_grid_whd']
    voxel_root = config.get('voxel_root', None)
    task = config.get('task', None)
    max_demos = config.get('max_demos', None)
    cache_suffix = config.get('cache_suffix', '')

    dataset = get_point_cloud_dataset(
        ds, root, mode=mode,
        voxelize=True,
        voxel_grid_whd=voxel_grid_whd,
        voxel_mode="occupancy",
        cache_dir=voxel_root,
        task=task,
        max_demos=max_demos,
        cache_suffix=cache_suffix,
    )

    print(f"[data] Loaded {mode} dataset: {len(dataset)} samples")
    return dataset


def export_examples(
    model,
    dataset,
    output_dir,
    indices=None,
    num_examples=10,
    iso_level=0.15,
    device='cuda',
    export_fg_bg=True,
):
    """Run inference and export examples for Blender rendering."""

    os.makedirs(output_dir, exist_ok=True)

    # Determine which indices to export
    if indices is not None:
        export_indices = indices
    else:
        # Sample evenly from dataset
        total = len(dataset)
        if num_examples >= total:
            export_indices = list(range(total))
        else:
            export_indices = np.linspace(0, total - 1, num_examples, dtype=int).tolist()

    print(f"[export] Exporting {len(export_indices)} examples to {output_dir}")

    exported = []

    for i, idx in enumerate(export_indices):
        print(f"\n[{i+1}/{len(export_indices)}] Processing example {idx}...")

        # Get data
        batch = dataset[idx]
        vox = batch["voxels"].unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            output = model(vox, warmup=False, with_loss=False)

        # Extract outputs
        gt_vol = output['x'][0].cpu()  # [C, D, H, W]
        rec_vol = output['rec'][0].cpu()

        # Get keypoints (in normalized [-1, 1] coords)
        z_base = output['z_base'][0].cpu()  # [K, 3]
        mu_offset = output['mu_offset'][0].cpu()
        mu_tot = z_base + mu_offset  # [K, 3]

        # Get obj_on scores for filtering (optional)
        obj_on = output.get('obj_on', None)
        if obj_on is not None:
            obj_on = obj_on[0].cpu().squeeze(-1)  # [K]

        # Create example directory
        example_dir = os.path.join(output_dir, f"example_{idx:04d}")

        # Export main meshes
        results = export_example_for_blender(
            output_dir=example_dir,
            gt_rgb_vol=gt_vol,
            rec_rgb_vol=rec_vol,
            keypoints=mu_tot,
            iso_level=iso_level,
            world_bounds=(-1, 1),
            example_name=f"example_{idx:04d}",
        )

        # Optionally export foreground and background separately
        if export_fg_bg:
            fg_vol = output.get('rgb_obj')
            bg_vol = output.get('bg_rgb')

            if fg_vol is not None:
                fg_path = os.path.join(example_dir, "fg.ply")
                from render.export_vox_to_mesh import export_voxel_field_to_ply
                try:
                    export_voxel_field_to_ply(
                        fg_vol[0].cpu(),
                        fg_path,
                        iso_level=iso_level,
                        world_bounds=(-1, 1),
                    )
                    results['fg_mesh'] = fg_path
                except Exception as e:
                    print(f"[warn] Failed to export fg mesh: {e}")

            if bg_vol is not None:
                bg_path = os.path.join(example_dir, "bg.ply")
                try:
                    export_voxel_field_to_ply(
                        bg_vol[0].cpu(),
                        bg_path,
                        iso_level=iso_level * 0.5,  # lower threshold for bg
                        world_bounds=(-1, 1),
                    )
                    results['bg_mesh'] = bg_path
                except Exception as e:
                    print(f"[warn] Failed to export bg mesh: {e}")

        # Save additional metadata
        meta_path = os.path.join(example_dir, "meta.json")
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        meta['dataset_idx'] = idx
        meta['num_keypoints'] = len(mu_tot)
        if obj_on is not None:
            meta['obj_on_mean'] = float(obj_on.mean())
            meta['obj_on_max'] = float(obj_on.max())

        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        exported.append(example_dir)
        print(f"    -> Exported to {example_dir}")

    print(f"\n[export] Done! Exported {len(exported)} examples")
    return exported


def main():
    parser = argparse.ArgumentParser(
        description="Export voxel reconstructions for Blender rendering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage - export 10 val examples
    python render/export_for_render.py --log_dir logs/240126_044143_mimicgen_rgb_coffee

    # Export more examples
    python render/export_for_render.py --log_dir logs/my_run --num_examples 50

    # Export specific indices
    python render/export_for_render.py --log_dir logs/my_run --indices 0 10 20 30

    # Use train set instead of val
    python render/export_for_render.py --log_dir logs/my_run --split train

    # Custom output directory
    python render/export_for_render.py --log_dir logs/my_run --output_dir ./renders/my_export
        """
    )

    parser.add_argument(
        "--log_dir", "-l", required=True,
        help="Path to log directory containing hparams.json and saves/best.pt"
    )
    parser.add_argument(
        "--ckpt", "-c", default="best.pt",
        help="Checkpoint filename (default: best.pt)"
    )
    parser.add_argument(
        "--output_dir", "-o", default=None,
        help="Output directory (default: {log_dir}/paper_meshes)"
    )
    parser.add_argument(
        "--num_examples", "-n", type=int, default=10,
        help="Number of examples to export (default: 10)"
    )
    parser.add_argument(
        "--indices", "-i", type=int, nargs="+", default=None,
        help="Specific dataset indices to export (overrides --num_examples)"
    )
    parser.add_argument(
        "--split", "-s", default="val", choices=["train", "val", "test"],
        help="Dataset split to use (default: val)"
    )
    parser.add_argument(
        "--iso_level", type=float, default=0.15,
        help="Marching cubes iso level (default: 0.15)"
    )
    parser.add_argument(
        "--device", "-d", default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--no_fg_bg", action="store_true",
        help="Don't export separate foreground/background meshes"
    )

    args = parser.parse_args()

    # Resolve paths
    log_dir = os.path.abspath(args.log_dir)
    ckpt_path = os.path.join(log_dir, "saves", args.ckpt)
    output_dir = args.output_dir or os.path.join(log_dir, "paper_meshes")

    # Validate paths
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"Log directory not found: {log_dir}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print("=" * 60)
    print("Export for Blender Rendering")
    print("=" * 60)
    print(f"Log dir:      {log_dir}")
    print(f"Checkpoint:   {ckpt_path}")
    print(f"Output dir:   {output_dir}")
    print(f"Split:        {args.split}")
    print(f"Num examples: {args.num_examples if args.indices is None else len(args.indices)}")
    print(f"ISO level:    {args.iso_level}")
    print(f"Device:       {args.device}")
    print("=" * 60)

    # Load config
    config = load_config(log_dir)

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_model(config, ckpt_path, device)

    # Load dataset
    dataset = load_dataset(config, mode=args.split)

    # Export examples
    exported = export_examples(
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        indices=args.indices,
        num_examples=args.num_examples,
        iso_level=args.iso_level,
        device=device,
        export_fg_bg=not args.no_fg_bg,
    )

    # Print next steps
    print("\n" + "=" * 60)
    print("Next Steps")
    print("=" * 60)
    print(f"\n1. Render with Blender:")
    print(f"   blender -b -P render/render_blender.py -- \\")
    print(f"       -i {exported[0] if exported else output_dir + '/example_0000'} \\")
    print(f"       -o renders/example_0000 \\")
    print(f"       --preset vertex_color --frames 60")

    print(f"\n2. Or batch render all:")
    print(f"   ./render/batch_render.sh {output_dir} renders/ --preset vertex_color")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
