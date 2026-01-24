# render/__init__.py
"""
Rendering utilities for voxel-to-mesh export and Blender visualization.

Main modules:
    - export_vox_to_mesh: Export voxel volumes to PLY meshes
    - render_blender: Blender Python script for rendering (run from Blender)

Example usage:
    from render.export_vox_to_mesh import (
        export_voxel_field_to_ply,
        export_keypoints_to_json,
        export_example_for_blender,
    )

    # Export a single example
    export_example_for_blender(
        output_dir="./paper_meshes/example_000",
        gt_rgb_vol=gt_vol,        # [3, D, H, W]
        rec_rgb_vol=rec_vol,      # [3, D, H, W]
        keypoints=mu_tot_b0,      # [K, 3]
        iso_level=0.15,
    )

Then render with Blender:
    blender -b -P render/render_blender.py -- \\
        --input_dir ./paper_meshes/example_000 \\
        --output_dir ./renders/example_000 \\
        --preset paper

Or use batch rendering:
    ./render/batch_render.sh ./paper_meshes ./renders --preset paper --frames 120
"""

from .export_vox_to_mesh import (
    export_voxel_field_to_ply,
    export_keypoints_to_json,
    export_meta_json,
    export_example_for_blender,
    compute_occupancy_from_rgb,
)

__all__ = [
    "export_voxel_field_to_ply",
    "export_keypoints_to_json",
    "export_meta_json",
    "export_example_for_blender",
    "compute_occupancy_from_rgb",
]
