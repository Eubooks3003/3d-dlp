#!/usr/bin/env python3
"""
render_blender.py

Blender Python script for rendering PLY meshes exported from voxel models.
Produces paper-quality stills and turntable videos with consistent camera/lighting.

Usage:
    blender -b -P render_blender.py -- \\
        --input_dir /path/to/example_000 \\
        --output_dir /path/to/renders \\
        --preset paper \\
        --engine CYCLES \\
        --device GPU \\
        --frames 120 \\
        --res 1920 1080

Prerequisites:
    - Blender 3.0+ (tested with 3.6)
    - Input directory containing: gt.ply, rec.ply, kp.json, meta.json

The script will render:
    - Static views: iso, front, side, top
    - Turntable animation frames
    - Optional: keypoints overlay
"""

import os
import sys
import json
import math
import argparse

# This script is run inside Blender's Python environment
try:
    import bpy
    import bmesh
    from mathutils import Vector, Matrix, Euler
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False
    print("[ERROR] This script must be run from within Blender:")
    print("  blender -b -P render_blender.py -- [args]")
    sys.exit(1)


# =============================================================================
# Configuration Presets
# =============================================================================

PRESETS = {
    "paper": {
        # Clean, professional look for publications
        "background_color": (0.95, 0.95, 0.95, 1.0),  # Light gray
        "floor_color": (0.85, 0.85, 0.85, 1.0),
        "material_type": "clay",  # Clay material for geometry
        "clay_color": (0.75, 0.75, 0.75, 1.0),
        "clay_roughness": 0.5,
        "use_hdri": False,
        "key_light_energy": 800,
        "fill_light_energy": 300,
        "rim_light_energy": 400,
        "samples": 256,
        "use_denoiser": True,
        "use_dof": False,
        "shadow_softness": 0.3,
    },
    "web": {
        # More dramatic look for website/demos
        "background_color": (0.1, 0.1, 0.12, 1.0),  # Dark
        "floor_color": (0.15, 0.15, 0.17, 1.0),
        "material_type": "vertex_color",  # Use actual vertex colors
        "clay_color": (0.8, 0.8, 0.8, 1.0),
        "clay_roughness": 0.3,
        "use_hdri": False,
        "key_light_energy": 1000,
        "fill_light_energy": 400,
        "rim_light_energy": 600,
        "samples": 512,
        "use_denoiser": True,
        "use_dof": False,  # Set True for hero shots
        "shadow_softness": 0.2,
    },
    "vertex_color": {
        # Show actual vertex colors from PLY
        "background_color": (1.0, 1.0, 1.0, 1.0),  # White
        "floor_color": (0.9, 0.9, 0.9, 1.0),
        "material_type": "vertex_color",
        "clay_color": None,
        "clay_roughness": 0.4,
        "use_hdri": False,
        "key_light_energy": 600,
        "fill_light_energy": 300,
        "rim_light_energy": 300,
        "samples": 256,
        "use_denoiser": True,
        "use_dof": False,
        "shadow_softness": 0.3,
    },
}

# Camera view definitions (elevation_deg, azimuth_deg, distance_mult)
CAMERA_VIEWS = {
    "iso": (30, 45, 1.0),       # Classic isometric-ish view
    "front": (0, 0, 1.0),       # Front view
    "side": (0, 90, 1.0),       # Side view
    "top": (89, 0, 1.0),        # Top-down view
    "three_quarter": (25, 30, 1.0),
}


# =============================================================================
# Scene Setup Functions
# =============================================================================

def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Also clear orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def setup_render_settings(engine="CYCLES", device="GPU", samples=256, res=(1920, 1080), use_denoiser=True):
    """Configure render settings."""
    scene = bpy.context.scene

    # Set engine
    if engine.upper() == "CYCLES":
        scene.render.engine = 'CYCLES'
        scene.cycles.device = device.upper()

        # Enable GPU if requested
        if device.upper() == "GPU":
            prefs = bpy.context.preferences.addons['cycles'].preferences
            prefs.compute_device_type = 'CUDA'  # or 'OPTIX', 'HIP', 'METAL'
            for dev in prefs.devices:
                dev.use = True

        scene.cycles.samples = samples

        # Denoiser
        if use_denoiser:
            scene.cycles.use_denoising = True
            # Try to set denoiser, but handle missing denoisers gracefully
            try:
                scene.cycles.denoiser = 'OPENIMAGEDENOISE'
            except TypeError:
                try:
                    scene.cycles.denoiser = 'NLM'  # Fallback to NLM
                except TypeError:
                    print("Warning: No denoiser available, disabling denoising")
                    scene.cycles.use_denoising = False
    else:
        scene.render.engine = 'BLENDER_EEVEE'
        scene.eevee.taa_render_samples = samples

    # Resolution
    scene.render.resolution_x = res[0]
    scene.render.resolution_y = res[1]
    scene.render.resolution_percentage = 100

    # Output settings
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = True  # Transparent background option


def setup_world_background(color=(0.95, 0.95, 0.95, 1.0)):
    """Set up world background color."""
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()

    # Background node
    bg_node = nodes.new('ShaderNodeBackground')
    bg_node.inputs['Color'].default_value = color
    bg_node.inputs['Strength'].default_value = 1.0

    # Output
    output_node = nodes.new('ShaderNodeOutputWorld')

    world.node_tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])


def create_floor_plane(size=10, color=(0.85, 0.85, 0.85, 1.0), z_offset=-0.01):
    """Create a floor plane for shadow catching."""
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, z_offset))
    floor = bpy.context.active_object
    floor.name = "Floor"

    # Create material
    mat = bpy.data.materials.new("FloorMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    # Principled BSDF
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Roughness'].default_value = 0.8
    if 'Specular IOR Level' in bsdf.inputs:
        bsdf.inputs['Specular IOR Level'].default_value = 0.1
    elif 'Specular' in bsdf.inputs:
        bsdf.inputs['Specular'].default_value = 0.1

    # Output
    output = nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    floor.data.materials.append(mat)

    return floor


def setup_three_point_lighting(key_energy=800, fill_energy=300, rim_energy=400, shadow_softness=0.3):
    """Create classic three-point lighting setup."""
    lights = []

    # Key light (main light, slightly above and to the side)
    bpy.ops.object.light_add(type='AREA', location=(3, -2, 4))
    key = bpy.context.active_object
    key.name = "KeyLight"
    key.data.energy = key_energy
    key.data.size = 2.0 * shadow_softness + 0.5
    key.rotation_euler = (math.radians(45), 0, math.radians(30))
    lights.append(key)

    # Fill light (softer, opposite side)
    bpy.ops.object.light_add(type='AREA', location=(-3, -1, 2))
    fill = bpy.context.active_object
    fill.name = "FillLight"
    fill.data.energy = fill_energy
    fill.data.size = 3.0
    fill.rotation_euler = (math.radians(60), 0, math.radians(-45))
    lights.append(fill)

    # Rim light (behind and above, for edge definition)
    bpy.ops.object.light_add(type='AREA', location=(0, 3, 3))
    rim = bpy.context.active_object
    rim.name = "RimLight"
    rim.data.energy = rim_energy
    rim.data.size = 2.0
    rim.rotation_euler = (math.radians(135), 0, 0)
    lights.append(rim)

    return lights


def setup_camera(distance=3.0, focal_length=50):
    """Create and configure camera."""
    bpy.ops.object.camera_add(location=(0, -distance, 0))
    camera = bpy.context.active_object
    camera.name = "Camera"
    camera.data.lens = focal_length
    camera.data.clip_start = 0.01
    camera.data.clip_end = 100

    # Set as active camera
    bpy.context.scene.camera = camera

    return camera


def position_camera_for_view(camera, target_center, bounding_radius, view_name="iso"):
    """Position camera for a specific view."""
    elev_deg, azim_deg, dist_mult = CAMERA_VIEWS.get(view_name, (30, 45, 1.0))

    # Calculate distance based on bounding radius and camera FOV
    fov_rad = camera.data.angle
    distance = (bounding_radius * 2.5 * dist_mult) / math.tan(fov_rad / 2)
    distance = max(distance, bounding_radius * 2)

    # Convert angles to radians
    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)

    # Calculate camera position
    x = target_center[0] + distance * math.cos(elev) * math.sin(azim)
    y = target_center[1] - distance * math.cos(elev) * math.cos(azim)
    z = target_center[2] + distance * math.sin(elev)

    camera.location = Vector((x, y, z))

    # Point camera at target
    direction = Vector(target_center) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()


def position_camera_for_turntable(camera, target_center, bounding_radius, frame, total_frames, elevation_deg=25):
    """Position camera for turntable animation frame."""
    azim_deg = (frame / total_frames) * 360

    # Calculate distance
    fov_rad = camera.data.angle
    distance = (bounding_radius * 2.5) / math.tan(fov_rad / 2)
    distance = max(distance, bounding_radius * 2)

    elev = math.radians(elevation_deg)
    azim = math.radians(azim_deg)

    x = target_center[0] + distance * math.cos(elev) * math.sin(azim)
    y = target_center[1] - distance * math.cos(elev) * math.cos(azim)
    z = target_center[2] + distance * math.sin(elev)

    camera.location = Vector((x, y, z))

    direction = Vector(target_center) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()


# =============================================================================
# Mesh Import and Processing
# =============================================================================

def import_ply(filepath, name="Mesh"):
    """Import a PLY file."""
    if not os.path.exists(filepath):
        print(f"[WARNING] PLY file not found: {filepath}")
        return None

    # Handle different Blender versions
    try:
        # Blender 3.4+
        bpy.ops.wm.ply_import(filepath=filepath)
    except AttributeError:
        # Blender 3.0 - 3.3
        bpy.ops.import_mesh.ply(filepath=filepath)
    obj = bpy.context.active_object
    if obj:
        obj.name = name
    return obj


def get_mesh_bounds(obj):
    """Get bounding box info for a mesh object."""
    if obj is None:
        return None, None, 0

    # Get world-space bounds
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_corner = Vector((
        min(c.x for c in bbox_corners),
        min(c.y for c in bbox_corners),
        min(c.z for c in bbox_corners),
    ))
    max_corner = Vector((
        max(c.x for c in bbox_corners),
        max(c.y for c in bbox_corners),
        max(c.z for c in bbox_corners),
    ))

    center = (min_corner + max_corner) / 2
    size = max_corner - min_corner
    radius = size.length / 2

    return center, size, radius


def normalize_mesh_transform(obj, target_size=2.0, center_at_origin=True):
    """Normalize mesh to fit within target_size, optionally center at origin."""
    if obj is None:
        return

    center, size, radius = get_mesh_bounds(obj)

    if radius < 1e-6:
        return

    # Scale to fit
    scale_factor = target_size / (radius * 2)
    obj.scale = Vector((scale_factor, scale_factor, scale_factor))

    # Apply scale
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(scale=True)

    # Re-center
    if center_at_origin:
        center, size, radius = get_mesh_bounds(obj)
        obj.location = -center

        # Apply location
        bpy.ops.object.transform_apply(location=True)


def create_clay_material(color=(0.75, 0.75, 0.75, 1.0), roughness=0.5, name="ClayMaterial"):
    """Create a simple clay/matte material."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Roughness'].default_value = roughness
    if 'Specular IOR Level' in bsdf.inputs:
        bsdf.inputs['Specular IOR Level'].default_value = 0.3
    elif 'Specular' in bsdf.inputs:
        bsdf.inputs['Specular'].default_value = 0.3

    output = nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    return mat


def create_vertex_color_material(roughness=0.4, name="VertexColorMaterial"):
    """Create material that uses vertex colors."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Vertex Color node
    vc_node = nodes.new('ShaderNodeVertexColor')
    vc_node.layer_name = "Col"  # Default vertex color layer name

    # Principled BSDF
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Roughness'].default_value = roughness
    # Handle different Blender versions
    if 'Specular IOR Level' in bsdf.inputs:
        bsdf.inputs['Specular IOR Level'].default_value = 0.3
    elif 'Specular' in bsdf.inputs:
        bsdf.inputs['Specular'].default_value = 0.3

    links.new(vc_node.outputs['Color'], bsdf.inputs['Base Color'])

    # Output
    output = nodes.new('ShaderNodeOutputMaterial')
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    return mat


def apply_material_to_mesh(obj, material):
    """Apply material to mesh object."""
    if obj is None or material is None:
        return

    # Clear existing materials
    obj.data.materials.clear()
    obj.data.materials.append(material)


# =============================================================================
# Keypoint Visualization
# =============================================================================

def load_keypoints_from_json(filepath):
    """Load keypoints from JSON file."""
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        data = json.load(f)

    return data.get("keypoints", [])


def create_keypoint_spheres(keypoints, radius=0.02, color=(1.0, 0.2, 0.2, 1.0)):
    """Create small spheres at keypoint locations."""
    if not keypoints:
        return []

    # Create material
    mat = bpy.data.materials.new("KeypointMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = color
        bsdf.inputs['Roughness'].default_value = 0.3
        # Make slightly emissive for visibility
        # Handle different Blender versions
        if 'Emission Color' in bsdf.inputs:
            # Blender 4.0+
            bsdf.inputs['Emission Color'].default_value = color
            bsdf.inputs['Emission Strength'].default_value = 0.5
        elif 'Emission' in bsdf.inputs:
            # Blender 3.x
            bsdf.inputs['Emission'].default_value = color

    spheres = []
    for i, kp in enumerate(keypoints):
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=radius,
            segments=16,
            ring_count=8,
            location=tuple(kp),
        )
        sphere = bpy.context.active_object
        sphere.name = f"KP_{i:03d}"
        sphere.data.materials.append(mat)
        spheres.append(sphere)

    return spheres


# =============================================================================
# Rendering Functions
# =============================================================================

def render_still(output_path, camera, target_center, bounding_radius, view_name="iso"):
    """Render a single still image."""
    position_camera_for_view(camera, target_center, bounding_radius, view_name)

    # Set output path
    bpy.context.scene.render.filepath = output_path

    # Render
    bpy.ops.render.render(write_still=True)
    print(f"[render] Saved: {output_path}")


def render_turntable_frames(output_dir, camera, target_center, bounding_radius, total_frames=120, elevation_deg=25):
    """Render turntable animation frames."""
    os.makedirs(output_dir, exist_ok=True)

    for frame in range(total_frames):
        position_camera_for_turntable(
            camera, target_center, bounding_radius,
            frame, total_frames, elevation_deg
        )

        output_path = os.path.join(output_dir, f"frame_{frame:04d}.png")
        bpy.context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)

        if frame % 10 == 0:
            print(f"[render] Turntable: {frame + 1}/{total_frames}")

    print(f"[render] Turntable complete: {total_frames} frames in {output_dir}")


# =============================================================================
# Main Pipeline
# =============================================================================

def render_example(
    input_dir,
    output_dir,
    preset="paper",
    engine="CYCLES",
    device="GPU",
    res=(1920, 1080),
    turntable_frames=120,
    render_gt=True,
    render_rec=True,
    render_keypoints=True,
    views=None,
):
    """
    Full rendering pipeline for a single example.

    Args:
        input_dir: directory containing gt.ply, rec.ply, kp.json, meta.json
        output_dir: directory to save renders
        preset: "paper" or "web"
        engine: "CYCLES" or "EEVEE"
        device: "GPU" or "CPU"
        res: (width, height) resolution
        turntable_frames: number of frames for turntable (0 to skip)
        render_gt: render ground truth mesh
        render_rec: render reconstruction mesh
        render_keypoints: include keypoint spheres
        views: list of view names to render (default: all)
    """
    if views is None:
        views = ["iso", "front", "side", "top"]

    config = PRESETS.get(preset, PRESETS["paper"])

    os.makedirs(output_dir, exist_ok=True)

    # Clear and set up scene
    clear_scene()
    setup_render_settings(engine, device, config["samples"], res, config["use_denoiser"])
    setup_world_background(config["background_color"])

    # Load meta if available
    meta_path = os.path.join(input_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)

    # Import meshes
    meshes_to_render = []

    if render_gt:
        gt_path = os.path.join(input_dir, "gt.ply")
        if os.path.exists(gt_path):
            gt_mesh = import_ply(gt_path, "GT_Mesh")
            if gt_mesh:
                normalize_mesh_transform(gt_mesh)
                meshes_to_render.append(("gt", gt_mesh))

    if render_rec:
        rec_path = os.path.join(input_dir, "rec.ply")
        if os.path.exists(rec_path):
            rec_mesh = import_ply(rec_path, "REC_Mesh")
            if rec_mesh:
                normalize_mesh_transform(rec_mesh)
                meshes_to_render.append(("rec", rec_mesh))

    if not meshes_to_render:
        print(f"[ERROR] No meshes found in {input_dir}")
        return

    # Load keypoints
    kp_spheres = []
    if render_keypoints:
        kp_path = os.path.join(input_dir, "kp.json")
        keypoints = load_keypoints_from_json(kp_path)
        if keypoints:
            # Scale keypoints to match normalized mesh
            # (assuming keypoints are in same coord system as mesh)
            kp_spheres = create_keypoint_spheres(keypoints, radius=0.03)

    # Create materials
    if config["material_type"] == "clay":
        material = create_clay_material(config["clay_color"], config["clay_roughness"])
    else:
        material = create_vertex_color_material(config["clay_roughness"])

    # Set up lighting and floor
    setup_three_point_lighting(
        config["key_light_energy"],
        config["fill_light_energy"],
        config["rim_light_energy"],
        config["shadow_softness"],
    )

    camera = setup_camera()

    # Render each mesh type separately
    for mesh_name, mesh_obj in meshes_to_render:
        # Hide all meshes first
        for _, m in meshes_to_render:
            m.hide_render = True

        # Show current mesh
        mesh_obj.hide_render = False
        apply_material_to_mesh(mesh_obj, material)

        # Get bounds for camera positioning
        center, size, radius = get_mesh_bounds(mesh_obj)
        center = center or Vector((0, 0, 0))
        radius = radius or 1.0

        # Create floor below mesh
        floor = create_floor_plane(size=10, color=config["floor_color"], z_offset=center.z - radius - 0.02)

        # Render still views
        for view_name in views:
            output_path = os.path.join(output_dir, f"{mesh_name}_{view_name}.png")
            render_still(output_path, camera, tuple(center), radius, view_name)

        # Render turntable
        if turntable_frames > 0:
            turntable_dir = os.path.join(output_dir, f"turntable_{mesh_name}")
            render_turntable_frames(turntable_dir, camera, tuple(center), radius, turntable_frames)

        # Clean up floor for next mesh
        bpy.data.objects.remove(floor)

    # Optionally render with keypoints overlay
    if kp_spheres and render_rec:
        # Show rec mesh with keypoints
        for _, m in meshes_to_render:
            m.hide_render = True

        for mesh_name, mesh_obj in meshes_to_render:
            if mesh_name == "rec":
                mesh_obj.hide_render = False
                center, size, radius = get_mesh_bounds(mesh_obj)
                center = center or Vector((0, 0, 0))
                radius = radius or 1.0

                # Show keypoint spheres
                for s in kp_spheres:
                    s.hide_render = False

                floor = create_floor_plane(size=10, color=config["floor_color"], z_offset=center.z - radius - 0.02)

                for view_name in views:
                    output_path = os.path.join(output_dir, f"rec_kp_{view_name}.png")
                    render_still(output_path, camera, tuple(center), radius, view_name)

                bpy.data.objects.remove(floor)
                break

    print(f"[render] Complete! Output: {output_dir}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    # Parse arguments after '--'
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Render PLY meshes for paper/web")

    parser.add_argument("--input_dir", "-i", required=True, help="Input directory with PLY files")
    parser.add_argument("--output_dir", "-o", required=True, help="Output directory for renders")
    parser.add_argument("--preset", "-p", default="paper", choices=["paper", "web", "vertex_color"],
                        help="Rendering preset")
    parser.add_argument("--engine", "-e", default="CYCLES", choices=["CYCLES", "EEVEE"],
                        help="Render engine")
    parser.add_argument("--device", "-d", default="GPU", choices=["GPU", "CPU"],
                        help="Compute device")
    parser.add_argument("--res", type=int, nargs=2, default=[1920, 1080],
                        help="Resolution (width height)")
    parser.add_argument("--frames", "-f", type=int, default=120,
                        help="Turntable frames (0 to skip)")
    parser.add_argument("--views", nargs="+", default=["iso", "front", "side", "top"],
                        help="Views to render")
    parser.add_argument("--no-gt", action="store_true", help="Skip GT mesh")
    parser.add_argument("--no-rec", action="store_true", help="Skip REC mesh")
    parser.add_argument("--no-kp", action="store_true", help="Skip keypoints")

    args = parser.parse_args(argv)

    render_example(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        preset=args.preset,
        engine=args.engine,
        device=args.device,
        res=tuple(args.res),
        turntable_frames=args.frames,
        render_gt=not args.no_gt,
        render_rec=not args.no_rec,
        render_keypoints=not args.no_kp,
        views=args.views,
    )


if __name__ == "__main__":
    main()
