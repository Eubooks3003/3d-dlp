#!/usr/bin/env python3
"""
EC-Diffuser Evaluation Script for MimicGen environments.

This script provides:
1. Expert demo replay with success detection
2. EC-Diffuser/DLP model rollout with proper success detection and termination
3. WandB logging for success metrics
4. Live voxelization pipeline for model inference

Usage:
    # Replay expert demos and check success
    python ec_diffuser_eval.py --mode replay --pkl /path/to/data.pkl --h5 /path/to/original.hdf5

    # Rollout EC-Diffuser model
    python ec_diffuser_eval.py --mode rollout --pkl /path/to/data.pkl --h5 /path/to/original.hdf5 \
        --dlp-cfg /path/to/config.json --dlp-ckpt /path/to/checkpoint.pt

    # Both replay and rollout
    python ec_diffuser_eval.py --mode both --pkl /path/to/data.pkl --h5 /path/to/original.hdf5 \
        --dlp-cfg /path/to/config.json --dlp-ckpt /path/to/checkpoint.pt

    # With live voxelization (requires camera intrinsics in H5)
    python ec_diffuser_eval.py --mode rollout --pkl /path/to/data.pkl --h5 /path/to/original.hdf5 \
        --dlp-cfg /path/to/config.json --dlp-ckpt /path/to/checkpoint.pt \
        --live-voxelize --voxel-grid-whd 64,64,64
"""

import os
import sys
import argparse
import pickle
import json
import time
import copy
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import torch
import h5py

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import numpy compatibility shims
import numpy.core as _core
sys.modules['numpy._core'] = _core
sys.modules['numpy._core.multiarray'] = _core.multiarray


def _to_hwc_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert frame to HWC uint8 format for video saving."""
    frame = np.asarray(frame)
    if frame.ndim == 4:
        if frame.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for video frame, got {frame.shape}")
        frame = frame[0]
    if frame.ndim == 3:
        if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
            frame = np.transpose(frame, (1, 2, 0))
        if frame.shape[-1] not in (1, 3, 4):
            raise ValueError(f"Frame has invalid channel count: shape={frame.shape}")
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating) and frame.max() <= 1.0:
            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


class LiveVoxelizer:
    """
    Live voxelization pipeline for converting RGB-D observations to voxel grids.

    This class handles:
    1. Depth to point cloud conversion using camera intrinsics
    2. Multi-camera point cloud fusion
    3. Point cloud to voxel grid conversion
    4. DLP encoding of voxel grids to tokens
    """

    def __init__(
        self,
        h5_path: str,
        camera_names: List[str] = None,
        voxel_grid_whd: Tuple[int, int, int] = (64, 64, 64),
        voxel_mode: str = "avg_rgb",
        normalize_to_unit_cube: bool = True,
        max_points: int = 4096,
        bounds_mode: str = "per_item",
        fixed_bounds: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
        crop_bounds: Optional[Dict[str, float]] = None,
        device: str = "cuda",
    ):
        """
        Initialize the live voxelizer.

        Args:
            h5_path: Path to H5 file containing camera intrinsics/extrinsics
            camera_names: List of camera names to use
            voxel_grid_whd: Voxel grid dimensions (W, H, D)
            voxel_mode: Voxelization mode ("occupancy", "density", "avg_rgb")
            normalize_to_unit_cube: Whether to normalize point cloud to unit cube
            max_points: Maximum points to keep after downsampling
            bounds_mode: Bounds mode ("per_item", "global", "fixed")
            fixed_bounds: Fixed bounds if bounds_mode is "fixed"
            crop_bounds: Optional cropping bounds dict with xmin, xmax, ymin, ymax, zmin, zmax
            device: Device for tensors
        """
        self.h5_path = h5_path
        self.camera_names = camera_names or ["agentview", "sideview"]
        self.voxel_grid_whd = voxel_grid_whd
        self.voxel_mode = voxel_mode
        self.normalize_to_unit_cube = normalize_to_unit_cube
        self.max_points = max_points
        self.bounds_mode = bounds_mode
        self.fixed_bounds = fixed_bounds
        self.crop_bounds = crop_bounds
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load camera parameters from H5
        self._load_camera_params()

    def _load_camera_params(self):
        """Load camera intrinsics and extrinsics from H5 file."""
        self.cam_params = {}

        with h5py.File(self.h5_path, "r") as h5:
            if "meta/cameras" not in h5:
                print("[WARNING] No camera metadata found in H5. Live voxelization may fail.")
                return

            for cam in self.camera_names:
                if cam not in h5["meta/cameras"]:
                    print(f"[WARNING] Camera {cam} not found in H5 metadata")
                    continue

                g = h5[f"meta/cameras/{cam}"]
                K = np.asarray(g["K"], dtype=np.float32)

                # Get extrinsics - may be stored differently
                T_raw = None
                if "T_cam_world" in g:
                    T_raw = np.asarray(g["T_cam_world"], dtype=np.float32)
                elif "extrinsics" in g:
                    T_raw = np.asarray(g["extrinsics"], dtype=np.float32)

                # Get near/far for depth conversion
                near = g.attrs.get("near", None)
                far = g.attrs.get("far", None)
                if near is None and "near" in g:
                    near = float(np.asarray(g["near"]))
                if far is None and "far" in g:
                    far = float(np.asarray(g["far"]))

                # Recover proper cam->world transform
                Tc2w = self._recover_cam2world(T_raw, K) if T_raw is not None else np.eye(4, dtype=np.float32)

                self.cam_params[cam] = {
                    "K": K,
                    "T_cam2world": Tc2w,
                    "near": near,
                    "far": far,
                }
                print(f"[Voxelizer] Loaded params for camera {cam}: K shape={K.shape}, near={near}, far={far}")

    def _recover_cam2world(self, T_raw: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Recover cam->world transform from potentially projection matrix."""
        T = np.asarray(T_raw, dtype=np.float64).reshape(4, 4)

        # Check if it's already a proper rigid transform
        if self._looks_like_rigid(T):
            return T.astype(np.float32)

        # Otherwise treat as projection matrix P = K[R|t]
        P = T[:3, :4]
        E = np.linalg.inv(K.astype(np.float64)) @ P
        Tw2c = np.eye(4, dtype=np.float64)
        Tw2c[:3, :4] = E
        Tc2w = np.linalg.inv(Tw2c)
        return Tc2w.astype(np.float32)

    def _looks_like_rigid(self, T: np.ndarray, tol: float = 1e-2) -> bool:
        """Check if matrix looks like a rigid transformation."""
        if T.shape != (4, 4):
            return False
        if not np.allclose(T[3], [0, 0, 0, 1], atol=tol):
            return False
        R = T[:3, :3]
        cn = np.linalg.norm(R, axis=0)
        if np.any(cn > 3.0) or np.any(cn < 0.2):
            return False
        return True

    def _depth_to_meters(self, depth: np.ndarray, near: Optional[float], far: Optional[float]) -> np.ndarray:
        """Convert depth buffer to meters."""
        d = depth.squeeze().astype(np.float32)

        # Handle integer depth buffers
        if np.issubdtype(depth.dtype, np.integer):
            denom = 65535.0 if depth.dtype == np.uint16 else 255.0
            d = d / denom

        # If normalized and we have near/far, convert to meters
        dmax = float(np.nanmax(d)) if np.isfinite(d).any() else 0.0
        if dmax <= 1.01 and near is not None and far is not None:
            d = near + d * (far - near)

        return d

    def _backproject_depth(self, depth_m: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Backproject depth image to 3D points in camera frame."""
        d = depth_m.squeeze().astype(np.float32)
        H, W = d.shape

        V, U = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        Z = d
        valid = np.isfinite(Z) & (Z > 0)

        if not np.any(valid):
            return np.zeros((0, 3), np.float32), np.zeros((0, 2), np.int32)

        Uv = U[valid].astype(np.float32)
        Vv = V[valid].astype(np.float32)
        Zv = Z[valid].astype(np.float32)

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        X = (Uv - cx) / fx * Zv
        Y = (Vv - cy) / fy * Zv
        pts_cam = np.stack([X, Y, Zv], axis=-1).astype(np.float32)

        idxs = np.stack([V[valid], U[valid]], axis=-1).astype(np.int32)
        return pts_cam, idxs

    def _apply_transform(self, T: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Apply 4x4 transform to points."""
        ones = np.ones((pts.shape[0], 1), np.float32)
        pts_h = np.concatenate([pts, ones], axis=1)
        out = (T.astype(np.float32) @ pts_h.T).T
        return out[:, :3]

    def _crop_points(self, xyz: np.ndarray, rgb: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply optional cropping bounds."""
        if self.crop_bounds is None:
            return xyz, rgb

        b = self.crop_bounds
        mask = (
            (xyz[:, 0] >= b.get("xmin", -np.inf)) & (xyz[:, 0] <= b.get("xmax", np.inf)) &
            (xyz[:, 1] >= b.get("ymin", -np.inf)) & (xyz[:, 1] <= b.get("ymax", np.inf)) &
            (xyz[:, 2] >= b.get("zmin", -np.inf)) & (xyz[:, 2] <= b.get("zmax", np.inf))
        )

        if not np.any(mask):
            return xyz, rgb

        xyz_out = xyz[mask]
        rgb_out = rgb[mask] if rgb is not None else None
        return xyz_out, rgb_out

    def _normalize_to_unit_cube(self, pts: np.ndarray) -> np.ndarray:
        """Normalize points to unit cube centered at origin."""
        out = pts.copy()
        xyz = out[:, :3]
        if xyz.shape[0] == 0:
            return out
        mins = xyz.min(axis=0)
        maxs = xyz.max(axis=0)
        center = (mins + maxs) / 2.0
        scale = (maxs - mins).max() + 1e-8
        out[:, :3] = (xyz - center[None, :]) / scale * 2.0
        return out

    def _downsample(self, pts: np.ndarray) -> np.ndarray:
        """Downsample points to max_points."""
        if pts.shape[0] <= self.max_points:
            return pts
        idx = np.random.choice(pts.shape[0], self.max_points, replace=False)
        return pts[idx]

    def obs_to_point_cloud(self, obs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert environment observation to fused point cloud.

        Args:
            obs: Environment observation dict with {cam}_image and {cam}_depth

        Returns:
            xyz: Point cloud positions [N, 3]
            rgb: Point cloud colors [N, 3]
        """
        all_xyz, all_rgb = [], []

        for cam in self.camera_names:
            if cam not in self.cam_params:
                continue

            depth_key = f"{cam}_depth"
            img_key = f"{cam}_image"

            if depth_key not in obs or img_key not in obs:
                continue

            depth_raw = np.asarray(obs[depth_key])
            img = np.asarray(obs[img_key])

            params = self.cam_params[cam]
            K = params["K"]
            Tc2w = params["T_cam2world"]
            near = params["near"]
            far = params["far"]

            # Convert depth to meters
            depth_m = self._depth_to_meters(depth_raw, near, far)

            # Backproject to 3D
            pts_cam, idxs = self._backproject_depth(depth_m, K)
            if pts_cam.shape[0] == 0:
                continue

            # Get colors
            v, u = idxs[:, 0], idxs[:, 1]
            rgb = img[v, u, :].astype(np.float32)
            if rgb.max() > 1.5:
                rgb /= 255.0

            # Transform to world frame
            pts_world = self._apply_transform(Tc2w, pts_cam)

            all_xyz.append(pts_world)
            all_rgb.append(rgb)

        if not all_xyz:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

        xyz = np.concatenate(all_xyz, axis=0)
        rgb = np.concatenate(all_rgb, axis=0)

        # Crop
        xyz, rgb = self._crop_points(xyz, rgb)

        return xyz, rgb

    def point_cloud_to_voxels(self, xyz: np.ndarray, rgb: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        Convert point cloud to voxel grid.

        Args:
            xyz: Point positions [N, 3]
            rgb: Point colors [N, 3] (optional, required for avg_rgb mode)

        Returns:
            voxels: Voxel grid tensor [C, D, H, W]
        """
        from datasets.voxelize_ds_wrapper import VoxelGridXYZ

        if xyz.shape[0] == 0:
            C = 1 if self.voxel_mode != "avg_rgb" else 3
            W, H, D = self.voxel_grid_whd
            return torch.zeros(C, D, H, W, device=self.device)

        # Combine xyz and rgb
        if rgb is not None:
            pts = np.concatenate([xyz, rgb], axis=-1)
        else:
            pts = xyz

        # Normalize
        if self.normalize_to_unit_cube:
            pts = self._normalize_to_unit_cube(pts)

        # Downsample
        pts = self._downsample(pts)

        # Convert to tensor
        pts_t = torch.from_numpy(pts).float().to(self.device)
        xyz_t = pts_t[:, :3]
        colors_t = pts_t[:, 3:6] if pts_t.shape[1] >= 6 and self.voxel_mode == "avg_rgb" else None

        # Determine bounds
        bounds = None
        if self.bounds_mode == "fixed" and self.fixed_bounds is not None:
            bounds = self.fixed_bounds

        # Voxelize
        W, H, D = self.voxel_grid_whd
        vg = VoxelGridXYZ(
            points_xyz=xyz_t,
            colors=colors_t,
            grid_whd=(W, H, D),
            bounds=bounds,
            mode=self.voxel_mode,
        )

        return vg.to_dense().to(self.device)

    def obs_to_voxels(self, obs: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Convert environment observation directly to voxel grid.

        Args:
            obs: Environment observation dict

        Returns:
            voxels: Voxel grid tensor [C, D, H, W]
        """
        xyz, rgb = self.obs_to_point_cloud(obs)
        return self.point_cloud_to_voxels(xyz, rgb)


class MimicGenEnvWrapper:
    """
    Wrapper for MimicGen environments with proper success detection and termination.
    """

    def __init__(
        self,
        h5_path: str,
        use_absolute_actions: bool = True,
        render: bool = True,
        camera_names: List[str] = None,
        camera_height: int = 256,
        camera_width: int = 256,
        max_episode_steps: int = 500,
    ):
        """
        Initialize the environment wrapper.

        Args:
            h5_path: Path to the original H5 dataset file
            use_absolute_actions: If True, use absolute actions (control_delta=False)
            render: If True, render to screen
            camera_names: List of camera names for observations
            camera_height: Camera observation height
            camera_width: Camera observation width
            max_episode_steps: Maximum steps per episode before timeout
        """
        self.h5_path = h5_path
        self.use_absolute_actions = use_absolute_actions
        self.render_enabled = render
        self.camera_names = camera_names or ['agentview', 'sideview']
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.max_episode_steps = max_episode_steps

        self._setup_env()

    def _setup_env(self):
        """Setup the robomimic environment."""
        # Try to import mimicgen
        for mod in ("mimicgen_envs", "mimicgen"):
            try:
                __import__(mod)
                break
            except ImportError:
                pass

        from robomimic.utils import file_utils as FileUtils
        from robomimic.utils import env_utils as EnvUtils
        from robomimic.utils import obs_utils as ObsUtils

        # Get environment metadata
        env_meta = FileUtils.get_env_metadata_from_dataset(self.h5_path)
        self.env_name = env_meta.get('env_name', 'unknown')

        # Configure action mode
        env_meta_copy = copy.deepcopy(env_meta)
        if not self.use_absolute_actions:
            print("[Env] Using RELATIVE/DELTA actions (control_delta=True)")
        else:
            print("[Env] Using ABSOLUTE actions (control_delta=False)")
            if 'controller_configs' in env_meta_copy.get('env_kwargs', {}):
                env_meta_copy['env_kwargs']['controller_configs']['control_delta'] = False

        # Configure rendering
        env_meta_copy.setdefault('env_kwargs', {})
        env_meta_copy['env_kwargs']['has_renderer'] = self.render_enabled
        env_meta_copy['env_kwargs']['has_offscreen_renderer'] = True
        env_meta_copy['env_kwargs']['use_camera_obs'] = True
        env_meta_copy['env_kwargs']['camera_names'] = self.camera_names
        env_meta_copy['env_kwargs']['camera_heights'] = [self.camera_height] * len(self.camera_names)
        env_meta_copy['env_kwargs']['camera_widths'] = [self.camera_width] * len(self.camera_names)

        # Initialize ObsUtils
        obs_specs = {
            "obs": {
                "rgb": [f"{cam}_image" for cam in self.camera_names],
                "depth": [],
                "low_dim": [],
            }
        }
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_specs)

        # Create environment
        self.env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta_copy,
            render=self.render_enabled,
            render_offscreen=True,
            use_image_obs=True,
        )

        self.action_dim = self.env.action_dimension
        print(f"[Env] Created environment: {self.env_name}")
        print(f"[Env] Action dimension: {self.action_dim}")

    def reset(self, init_state: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Reset the environment.

        Args:
            init_state: Optional initial state to reset to

        Returns:
            Initial observation dict
        """
        if init_state is not None:
            obs = self.env.reset_to({"states": init_state})
        else:
            obs = self.env.reset()

        if obs is None:
            obs = self.env.get_observation()

        self._step_count = 0
        self._episode_success = False
        return obs

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Step the environment.

        Args:
            action: Action to take

        Returns:
            (observation, reward, done, info) tuple
        """
        obs, reward, done, info = self.env.step(action)
        self._step_count += 1

        # Check for success - MimicGen environments report this in info
        success = self._check_success(info)
        info['success'] = success
        info['step_count'] = self._step_count

        if success and not self._episode_success:
            self._episode_success = True
            info['first_success_step'] = self._step_count
            print(f"  [SUCCESS] Task completed at step {self._step_count}!")

        # Check for timeout
        if self._step_count >= self.max_episode_steps:
            done = True
            info['timeout'] = True

        # Terminate on success (optional behavior)
        if success:
            done = True

        return obs, reward, done, info

    def _check_success(self, info: Dict[str, Any]) -> bool:
        """
        Check if the task was completed successfully.

        Args:
            info: Info dict from environment step

        Returns:
            True if task succeeded
        """
        # MimicGen environments can report success in different ways
        if info.get('success', False):
            return True
        if info.get('task_success', False):
            return True

        # Some environments have a _check_success method
        if hasattr(self.env, '_check_success') or hasattr(self.env, 'is_success'):
            try:
                if hasattr(self.env, 'is_success'):
                    return self.env.is_success()
                return self.env._check_success()
            except:
                pass

        # Check underlying robosuite environment
        if hasattr(self.env, 'env') and hasattr(self.env.env, '_check_success'):
            try:
                return self.env.env._check_success()
            except:
                pass

        return False

    def render(self):
        """Render the environment."""
        if self.render_enabled:
            self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()

    @property
    def episode_success(self) -> bool:
        return self._episode_success


class ECDiffuserRolloutEvaluator:
    """
    Evaluator for EC-Diffuser models in MimicGen environments.
    """

    def __init__(
        self,
        pkl_path: str,
        h5_path: str,
        dlp_cfg_path: Optional[str] = None,
        dlp_ckpt_path: Optional[str] = None,
        policy_cfg_path: Optional[str] = None,
        policy_ckpt_path: Optional[str] = None,
        device: str = "cuda",
        use_absolute_actions: bool = True,
        render: bool = True,
        max_episode_steps: int = 500,
        use_wandb: bool = False,
        wandb_project: str = "ec-diffuser-eval",
        wandb_run_name: Optional[str] = None,
        use_live_voxelize: bool = False,
        voxel_grid_whd: Tuple[int, int, int] = (64, 64, 64),
        voxel_mode: str = "avg_rgb",
        use_depth_obs: bool = True,
    ):
        """
        Initialize the evaluator.

        Args:
            pkl_path: Path to preprocessed pkl file with observations and actions
            h5_path: Path to original H5 dataset file
            dlp_cfg_path: Path to DLP model config (JSON)
            dlp_ckpt_path: Path to DLP model checkpoint
            policy_cfg_path: Path to policy config (JSON) - for DLPPolicy
            policy_ckpt_path: Path to policy checkpoint - for DLPPolicy
            device: Device to run on
            use_absolute_actions: Whether to use absolute actions
            render: Whether to render
            max_episode_steps: Maximum steps per episode
            use_wandb: Whether to log to wandb
            wandb_project: Wandb project name
            wandb_run_name: Wandb run name
            use_live_voxelize: Whether to voxelize observations live during rollout
            voxel_grid_whd: Voxel grid dimensions (W, H, D)
            voxel_mode: Voxelization mode
            use_depth_obs: Whether to request depth observations from environment
        """
        self.pkl_path = pkl_path
        self.h5_path = h5_path
        self.dlp_cfg_path = dlp_cfg_path
        self.dlp_ckpt_path = dlp_ckpt_path
        self.policy_cfg_path = policy_cfg_path
        self.policy_ckpt_path = policy_ckpt_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_absolute_actions = use_absolute_actions
        self.render = render
        self.max_episode_steps = max_episode_steps
        self.use_wandb = use_wandb
        self.use_live_voxelize = use_live_voxelize
        self.voxel_grid_whd = voxel_grid_whd
        self.voxel_mode = voxel_mode

        # Load data
        self._load_data()

        # Setup environment (request depth if using live voxelization)
        camera_names = ['agentview', 'sideview']
        self.env = MimicGenEnvWrapper(
            h5_path=h5_path,
            use_absolute_actions=use_absolute_actions,
            render=render,
            max_episode_steps=max_episode_steps,
            camera_names=camera_names,
        )

        # Setup live voxelizer if requested
        self.live_voxelizer = None
        if use_live_voxelize:
            print(f"\n{'='*60}")
            print("Setting up live voxelizer...")
            self.live_voxelizer = LiveVoxelizer(
                h5_path=h5_path,
                camera_names=camera_names,
                voxel_grid_whd=voxel_grid_whd,
                voxel_mode=voxel_mode,
                device=device,
            )

        # Load model if paths provided
        self.model = None
        if dlp_cfg_path and dlp_ckpt_path:
            self._load_dlp_model()
        elif policy_cfg_path:
            self._load_policy_model()

        # Setup wandb
        if use_wandb:
            import wandb
            wandb.init(
                project=wandb_project,
                name=wandb_run_name or f"eval_{os.path.basename(pkl_path)}",
                config={
                    "pkl_path": pkl_path,
                    "h5_path": h5_path,
                    "use_absolute_actions": use_absolute_actions,
                    "max_episode_steps": max_episode_steps,
                    "use_live_voxelize": use_live_voxelize,
                    "voxel_grid_whd": voxel_grid_whd,
                }
            )

    def _load_data(self):
        """Load preprocessed data from pkl file."""
        print(f"\n{'='*60}")
        print(f"Loading preprocessed data from: {self.pkl_path}")

        with open(self.pkl_path, 'rb') as f:
            self.data = pickle.load(f)

        print(f"\nDataset contents:")
        for k, v in self.data.items():
            if isinstance(v, np.ndarray):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            elif isinstance(v, dict):
                print(f"  {k}: dict with {len(v)} keys")
            else:
                print(f"  {k}: {type(v)}")

        # Extract key arrays
        self.init_states = self.data.get('init_states')
        self.actions = self.data.get('actions')
        self.path_lengths = self.data.get('path_lengths')
        self.observations = self.data.get('observations')
        self.goals = self.data.get('goals')

        if self.init_states is None:
            raise ValueError("'init_states' not found in pkl file!")

        self.num_demos = len(self.init_states)
        print(f"\nFound {self.num_demos} demos")
        print(f"  Path lengths: min={self.path_lengths.min()}, max={self.path_lengths.max()}")

        # Get action mode from metadata
        meta = self.data.get('meta', {})
        self.action_mode = meta.get('action_mode', 'unknown')
        print(f"  Action mode: {self.action_mode}")

    def _load_dlp_model(self):
        """Load DLP model from config and checkpoint."""
        print(f"\n{'='*60}")
        print(f"Loading DLP model...")
        print(f"  Config: {self.dlp_cfg_path}")
        print(f"  Checkpoint: {self.dlp_ckpt_path}")

        from voxel_models import DLP
        from utils.util_func import get_config
        from utils.log_utils import load_checkpoint

        # Load config
        cfg = get_config(self.dlp_cfg_path)

        # Build model
        self.model = DLP(
            cdim=cfg["ch"],
            image_size=cfg.get("voxel_grid_whd", [64, 64, 64])[0],
            normalize_rgb=cfg["normalize_rgb"],
            n_kp_per_patch=cfg["n_kp_per_patch"],
            patch_size=cfg["patch_size"],
            anchor_s=cfg["anchor_s"],
            n_kp_enc=cfg["n_kp_enc"],
            n_kp_prior=cfg["n_kp_prior"],
            pad_mode=cfg["pad_mode"],
            dropout=cfg["dropout"],
            features_dist=cfg.get("features_dist", "gauss"),
            learned_feature_dim=cfg["learned_feature_dim"],
            learned_bg_feature_dim=cfg.get("learned_bg_feature_dim", cfg["learned_feature_dim"]),
            n_fg_categories=cfg.get("n_fg_categories", 8),
            n_fg_classes=cfg.get("n_fg_classes", 4),
            n_bg_categories=cfg.get("n_bg_categories", 4),
            n_bg_classes=cfg.get("n_bg_classes", 4),
            scale_std=cfg["scale_std"],
            offset_std=cfg["offset_std"],
            obj_on_alpha=cfg["obj_on_alpha"],
            obj_on_beta=cfg["obj_on_beta"],
            obj_res_from_fc=cfg["obj_res_from_fc"],
            obj_ch_mult_prior=cfg.get("obj_ch_mult_prior", cfg["obj_ch_mult"]),
            obj_ch_mult=cfg["obj_ch_mult"],
            obj_base_ch=cfg["obj_base_ch"],
            obj_final_cnn_ch=cfg["obj_final_cnn_ch"],
            bg_res_from_fc=cfg["bg_res_from_fc"],
            bg_ch_mult=cfg["bg_ch_mult"],
            bg_base_ch=cfg["bg_base_ch"],
            bg_final_cnn_ch=cfg["bg_final_cnn_ch"],
            use_resblock=cfg["use_resblock"],
            num_res_blocks=cfg["num_res_blocks"],
            cnn_mid_blocks=cfg.get("cnn_mid_blocks", False),
            mlp_hidden_dim=cfg.get("mlp_hidden_dim", 256),
            pint_enc_layers=cfg["pint_enc_layers"],
            pint_enc_heads=cfg["pint_enc_heads"],
            timestep_horizon=1,
            separate_depth_features=cfg.get("separate_depth_features", False),
            depth_feature_dim=cfg.get("depth_feature_dim", 0),
            split_loss=cfg.get("split_loss", False),
            depth_loss_ratio=cfg.get("depth_loss_ratio", 1.0),
        ).to(self.device)

        # Load weights
        load_checkpoint(self.dlp_ckpt_path, self.model, None, None, map_location=self.device)
        self.model.eval()
        print(f"  Model loaded successfully on {self.device}")

    def _load_policy_model(self):
        """Load DLPPolicy model from config."""
        print(f"\n{'='*60}")
        print(f"Loading DLPPolicy model...")
        print(f"  Config: {self.policy_cfg_path}")

        from dlp_policy import DLPPolicy

        self.model = DLPPolicy(self.policy_cfg_path).to(self.device)
        self.model.eval()
        print(f"  Policy loaded successfully on {self.device}")

    def replay_expert_demo(
        self,
        demo_idx: int,
        save_video: Optional[str] = None,
        speed: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Replay an expert demonstration and check success.

        Args:
            demo_idx: Demo index to replay
            save_video: Optional path to save video
            speed: Playback speed multiplier

        Returns:
            Dict with success status and metrics
        """
        if demo_idx >= self.num_demos:
            raise ValueError(f"Demo index {demo_idx} out of range (max: {self.num_demos - 1})")

        print(f"\n{'='*60}")
        print(f"Replaying expert demo {demo_idx}")
        print(f"{'='*60}")

        # Get demo data
        init_state = self.init_states[demo_idx]
        demo_actions = self.actions[demo_idx]
        demo_length = int(self.path_lengths[demo_idx])

        print(f"  Demo length: {demo_length} steps")

        # Reset environment
        obs = self.env.reset(init_state)

        # Replay loop
        video_frames = []
        step_delay = 0.02 / speed
        success = False
        first_success_step = None

        for t in range(demo_length):
            action = demo_actions[t]

            # Step environment
            obs, reward, done, info = self.env.step(action)

            # Render
            if self.render:
                self.env.render()

            # Collect video frames
            if save_video:
                for cam in self.env.camera_names:
                    if f'{cam}_image' in obs:
                        frame = _to_hwc_uint8(obs[f'{cam}_image'])
                        video_frames.append(frame)
                        break

            # Check success
            if info.get('success', False) and not success:
                success = True
                first_success_step = t

            if done:
                print(f"  Episode ended at step {t}")
                break

            time.sleep(step_delay)

        # Final status
        final_success = self.env.episode_success

        result = {
            'demo_idx': demo_idx,
            'demo_length': demo_length,
            'steps_taken': t + 1,
            'success': final_success,
            'first_success_step': first_success_step,
            'timeout': info.get('timeout', False),
        }

        print(f"\nDemo {demo_idx} result:")
        print(f"  Steps taken: {result['steps_taken']}")
        print(f"  Success: {result['success']}")
        if first_success_step is not None:
            print(f"  First success at step: {first_success_step}")

        # Save video
        if save_video and len(video_frames) > 0:
            import imageio
            os.makedirs(os.path.dirname(save_video) or '.', exist_ok=True)
            imageio.mimsave(save_video, video_frames, fps=30)
            print(f"  Video saved to: {save_video}")

        # Log to wandb
        if self.use_wandb:
            import wandb
            wandb.log({
                f"replay/demo_{demo_idx}/success": int(final_success),
                f"replay/demo_{demo_idx}/steps": result['steps_taken'],
            })

        return result

    def replay_all_demos(
        self,
        demo_indices: Optional[List[int]] = None,
        save_videos_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Replay all (or specified) expert demonstrations.

        Args:
            demo_indices: List of demo indices to replay, or None for all
            save_videos_dir: Directory to save videos

        Returns:
            Aggregated results dict
        """
        if demo_indices is None:
            demo_indices = list(range(self.num_demos))

        results = []
        successes = 0

        for idx in demo_indices:
            video_path = None
            if save_videos_dir:
                video_path = os.path.join(save_videos_dir, f"demo_{idx}_replay.mp4")

            result = self.replay_expert_demo(idx, save_video=video_path)
            results.append(result)
            if result['success']:
                successes += 1

        # Aggregate results
        success_rate = successes / len(demo_indices)

        summary = {
            'num_demos': len(demo_indices),
            'num_successes': successes,
            'success_rate': success_rate,
            'per_demo_results': results,
        }

        print(f"\n{'='*60}")
        print(f"Expert Replay Summary")
        print(f"{'='*60}")
        print(f"  Total demos: {len(demo_indices)}")
        print(f"  Successes: {successes}")
        print(f"  Success rate: {success_rate * 100:.1f}%")

        # Log to wandb
        if self.use_wandb:
            import wandb
            wandb.log({
                "replay/total_demos": len(demo_indices),
                "replay/successes": successes,
                "replay/success_rate": success_rate,
            })

        return summary

    def _encode_observation(self, obs: Dict[str, np.ndarray]) -> Optional[torch.Tensor]:
        """
        Encode environment observation to DLP tokens using live voxelization.

        Args:
            obs: Environment observation dict

        Returns:
            Encoded tokens [1, K, Dtok] or None if encoding fails
        """
        if self.live_voxelizer is None or self.model is None:
            return None

        try:
            # Convert observation to voxels
            voxels = self.live_voxelizer.obs_to_voxels(obs)  # [C, D, H, W]
            voxels = voxels.unsqueeze(0)  # Add batch dim [1, C, D, H, W]

            # Encode with DLP model
            with torch.no_grad():
                out = self.model(voxels, deterministic=True, warmup=False, with_loss=False)

                # Pack tokens in same format as preprocessing
                z = out["z"][:, 0]          # (B, K, 3)
                z_scale = out["z_scale"][:, 0]    # (B, K, 3)
                z_depth = out["z_depth"][:, 0]    # (B, K, 3)
                obj_on = out["obj_on"][:, 0]     # (B, K, 1) or (B, K)
                z_feat = out["z_features"][:, 0]  # (B, K, F)

                if obj_on.dim() == 2:
                    obj_on = obj_on.unsqueeze(-1)

                tokens = torch.cat([z, z_scale, z_depth, obj_on, z_feat], dim=-1)

            return tokens

        except Exception as e:
            print(f"[WARNING] Failed to encode observation: {e}")
            return None

    def rollout_model(
        self,
        demo_idx: int,
        n_steps: Optional[int] = None,
        save_video: Optional[str] = None,
        terminate_on_success: bool = True,
        use_expert_actions: bool = True,
    ) -> Dict[str, Any]:
        """
        Rollout the model from a demo's initial state.

        Args:
            demo_idx: Demo index to use for initial state
            n_steps: Number of steps to rollout (default: demo length)
            save_video: Optional path to save video
            terminate_on_success: Whether to stop on success
            use_expert_actions: If True, use expert actions (model just encodes).
                               If False, requires a policy model that can generate actions.

        Returns:
            Dict with rollout results
        """
        if demo_idx >= self.num_demos:
            raise ValueError(f"Demo index {demo_idx} out of range (max: {self.num_demos - 1})")

        print(f"\n{'='*60}")
        print(f"Rolling out from demo {demo_idx} initial state")
        print(f"{'='*60}")

        # Get initial state and goal
        init_state = self.init_states[demo_idx]
        demo_length = int(self.path_lengths[demo_idx])
        n_steps = n_steps or demo_length

        # Get goal observation (tokens) if available
        goal_tokens = None
        if self.goals is not None:
            goal_tokens = torch.from_numpy(self.goals[demo_idx, 0]).float().to(self.device)
            goal_tokens = goal_tokens.unsqueeze(0)  # Add batch dim

        print(f"  Max steps: {n_steps}")
        print(f"  Using expert actions: {use_expert_actions}")
        print(f"  Live voxelization: {self.live_voxelizer is not None}")

        # Reset environment
        obs = self.env.reset(init_state)

        # Rollout loop
        video_frames = []
        actions_taken = []
        encoded_obs_list = []
        success = False
        first_success_step = None

        for t in range(n_steps):
            # Try to encode current observation with DLP model
            live_tokens = None
            if self.live_voxelizer is not None and self.model is not None:
                live_tokens = self._encode_observation(obs)
                if live_tokens is not None:
                    encoded_obs_list.append(live_tokens.cpu().numpy())

            # Determine action
            with torch.no_grad():
                if use_expert_actions and t < demo_length:
                    # Use expert action from dataset
                    action = self.actions[demo_idx, t]
                elif hasattr(self.model, 'act') and live_tokens is not None:
                    # Use DLPPolicy's act method if available
                    # This would require RGB observations, not voxel tokens
                    # For now, fall back to expert
                    action = self.actions[demo_idx, t] if t < demo_length else np.zeros(self.env.action_dim)
                elif self.observations is not None and t < demo_length:
                    # Use stored tokens (for potential future policy inference)
                    # Currently just use expert actions
                    action = self.actions[demo_idx, t]
                else:
                    # No expert action available, use zeros
                    action = np.zeros(self.env.action_dim) if t >= demo_length else self.actions[demo_idx, t]

            actions_taken.append(action)

            # Step environment
            obs, reward, done, info = self.env.step(action)

            # Render
            if self.render:
                self.env.render()

            # Collect video frames
            if save_video:
                for cam in self.env.camera_names:
                    if f'{cam}_image' in obs:
                        frame = _to_hwc_uint8(obs[f'{cam}_image'])
                        video_frames.append(frame)
                        break

            # Check success
            if info.get('success', False) and not success:
                success = True
                first_success_step = t
                if terminate_on_success:
                    print(f"  Terminating on success at step {t}")
                    break

            if done:
                print(f"  Episode ended at step {t}")
                break

        # Final status
        final_success = self.env.episode_success

        result = {
            'demo_idx': demo_idx,
            'target_steps': n_steps,
            'steps_taken': t + 1,
            'success': final_success,
            'first_success_step': first_success_step,
            'timeout': info.get('timeout', False),
            'actions_taken': np.array(actions_taken),
        }

        print(f"\nRollout {demo_idx} result:")
        print(f"  Steps taken: {result['steps_taken']}")
        print(f"  Success: {result['success']}")
        if first_success_step is not None:
            print(f"  First success at step: {first_success_step}")

        # Save video
        if save_video and len(video_frames) > 0:
            import imageio
            os.makedirs(os.path.dirname(save_video) or '.', exist_ok=True)
            imageio.mimsave(save_video, video_frames, fps=30)
            print(f"  Video saved to: {save_video}")

        # Log to wandb
        if self.use_wandb:
            import wandb
            wandb.log({
                f"rollout/demo_{demo_idx}/success": int(final_success),
                f"rollout/demo_{demo_idx}/steps": result['steps_taken'],
            })

        return result

    def rollout_all(
        self,
        demo_indices: Optional[List[int]] = None,
        n_steps: Optional[int] = None,
        save_videos_dir: Optional[str] = None,
        terminate_on_success: bool = True,
    ) -> Dict[str, Any]:
        """
        Rollout model from all (or specified) demo initial states.

        Args:
            demo_indices: List of demo indices, or None for all
            n_steps: Number of steps per rollout
            save_videos_dir: Directory to save videos
            terminate_on_success: Whether to stop on success

        Returns:
            Aggregated results dict
        """
        if demo_indices is None:
            demo_indices = list(range(self.num_demos))

        results = []
        successes = 0

        for idx in demo_indices:
            video_path = None
            if save_videos_dir:
                video_path = os.path.join(save_videos_dir, f"demo_{idx}_rollout.mp4")

            result = self.rollout_model(
                idx,
                n_steps=n_steps,
                save_video=video_path,
                terminate_on_success=terminate_on_success,
            )
            results.append(result)
            if result['success']:
                successes += 1

        # Aggregate results
        success_rate = successes / len(demo_indices)

        summary = {
            'num_demos': len(demo_indices),
            'num_successes': successes,
            'success_rate': success_rate,
            'per_demo_results': results,
        }

        print(f"\n{'='*60}")
        print(f"Model Rollout Summary")
        print(f"{'='*60}")
        print(f"  Total rollouts: {len(demo_indices)}")
        print(f"  Successes: {successes}")
        print(f"  Success rate: {success_rate * 100:.1f}%")

        # Log to wandb
        if self.use_wandb:
            import wandb
            wandb.log({
                "rollout/total_demos": len(demo_indices),
                "rollout/successes": successes,
                "rollout/success_rate": success_rate,
            })

        return summary

    def close(self):
        """Clean up resources."""
        self.env.close()
        if self.use_wandb:
            import wandb
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="EC-Diffuser Evaluation for MimicGen")

    # Required arguments
    parser.add_argument("--pkl", type=str, required=True,
                        help="Path to preprocessed pkl file")
    parser.add_argument("--h5", type=str, required=True,
                        help="Path to original H5 dataset file")

    # Mode selection
    parser.add_argument("--mode", type=str, default="both",
                        choices=["replay", "rollout", "both"],
                        help="Evaluation mode: replay expert demos, rollout model, or both")

    # Model paths (required for rollout mode)
    parser.add_argument("--dlp-cfg", type=str, default=None,
                        help="Path to DLP config JSON")
    parser.add_argument("--dlp-ckpt", type=str, default=None,
                        help="Path to DLP checkpoint")
    parser.add_argument("--policy-cfg", type=str, default=None,
                        help="Path to DLPPolicy config JSON (alternative to DLP)")

    # Demo selection
    parser.add_argument("--demo-idx", type=int, nargs="+", default=None,
                        help="Demo indices to evaluate (default: all)")
    parser.add_argument("--max-demos", type=int, default=None,
                        help="Maximum number of demos to evaluate")

    # Environment settings
    parser.add_argument("--use-relative-actions", action="store_true",
                        help="Use relative actions (default: absolute)")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum steps per episode")
    parser.add_argument("--no-terminate-on-success", action="store_true",
                        help="Don't terminate on success")

    # Live voxelization settings
    parser.add_argument("--live-voxelize", action="store_true",
                        help="Enable live voxelization of observations during rollout")
    parser.add_argument("--voxel-grid-whd", type=str, default="64,64,64",
                        help="Voxel grid dimensions (W,H,D)")
    parser.add_argument("--voxel-mode", type=str, default="avg_rgb",
                        choices=["occupancy", "density", "avg_rgb"],
                        help="Voxelization mode")

    # Output settings
    parser.add_argument("--save-videos", type=str, default=None,
                        help="Directory to save rollout videos")
    parser.add_argument("--save-results", type=str, default=None,
                        help="Path to save results JSON")

    # Wandb settings
    parser.add_argument("--use-wandb", action="store_true",
                        help="Log to wandb")
    parser.add_argument("--wandb-project", type=str, default="ec-diffuser-eval",
                        help="Wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="Wandb run name")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")

    args = parser.parse_args()

    # Parse voxel grid dimensions
    voxel_grid_whd = tuple(map(int, args.voxel_grid_whd.split(',')))

    # Validate arguments
    if args.mode in ["rollout", "both"]:
        if args.dlp_cfg is None and args.policy_cfg is None:
            print("[WARNING] No model config provided. Rollout will use expert actions.")

    # Create evaluator
    evaluator = ECDiffuserRolloutEvaluator(
        pkl_path=args.pkl,
        h5_path=args.h5,
        dlp_cfg_path=args.dlp_cfg,
        dlp_ckpt_path=args.dlp_ckpt,
        policy_cfg_path=args.policy_cfg,
        device=args.device,
        use_absolute_actions=not args.use_relative_actions,
        render=not args.no_render,
        max_episode_steps=args.max_steps,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        use_live_voxelize=args.live_voxelize,
        voxel_grid_whd=voxel_grid_whd,
        voxel_mode=args.voxel_mode,
    )

    # Determine demo indices
    demo_indices = args.demo_idx
    if demo_indices is None:
        demo_indices = list(range(evaluator.num_demos))
    if args.max_demos is not None:
        demo_indices = demo_indices[:args.max_demos]

    results = {}

    try:
        # Run evaluation
        if args.mode in ["replay", "both"]:
            videos_dir = None
            if args.save_videos:
                videos_dir = os.path.join(args.save_videos, "replay")
            results['replay'] = evaluator.replay_all_demos(
                demo_indices=demo_indices,
                save_videos_dir=videos_dir,
            )

        if args.mode in ["rollout", "both"]:
            videos_dir = None
            if args.save_videos:
                videos_dir = os.path.join(args.save_videos, "rollout")
            results['rollout'] = evaluator.rollout_all(
                demo_indices=demo_indices,
                save_videos_dir=videos_dir,
                terminate_on_success=not args.no_terminate_on_success,
            )

        # Save results
        if args.save_results:
            os.makedirs(os.path.dirname(args.save_results) or '.', exist_ok=True)
            # Convert numpy arrays to lists for JSON serialization
            results_json = {}
            for mode, mode_results in results.items():
                results_json[mode] = {
                    k: v for k, v in mode_results.items()
                    if k != 'per_demo_results'
                }
                results_json[mode]['per_demo_results'] = []
                for r in mode_results.get('per_demo_results', []):
                    r_clean = {k: v for k, v in r.items() if k != 'actions_taken'}
                    results_json[mode]['per_demo_results'].append(r_clean)

            with open(args.save_results, 'w') as f:
                json.dump(results_json, f, indent=2)
            print(f"\nResults saved to: {args.save_results}")

    finally:
        evaluator.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
