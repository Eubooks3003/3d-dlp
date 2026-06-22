import numpy as np
import torch
import h5py

from datasets.voxelize_ds_wrapper import VoxelGridXYZ
from dlp_utils import log_rgb_voxels

# ----------------------------
# Gripper state extraction (matches ec_diffuser_mimicgen_preprocess.py)
# ----------------------------

def quat_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (x,y,z,w) to 6D rotation representation.
    6D = first two columns of rotation matrix, flattened.
    """
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # Rotation matrix from quaternion
    # First column
    r00 = 1 - 2*(y*y + z*z)
    r10 = 2*(x*y + w*z)
    r20 = 2*(x*z - w*y)

    # Second column
    r01 = 2*(x*y - w*z)
    r11 = 1 - 2*(x*x + z*z)
    r21 = 2*(y*z + w*x)

    # Stack first two columns as 6D representation
    rot6d = np.stack([r00, r10, r20, r01, r11, r21], axis=-1)
    return rot6d.astype(np.float32)


def extract_gripper_state_from_obs(raw_obs: dict) -> np.ndarray:
    """
    Extract gripper state from robosuite raw observation dict.

    Gripper state format: [pos(3), rot_6d(6), gripper_open(1)] = 10 dims

    Returns:
        gripper_state: [10,] array
    """
    # End-effector position (3D)
    eef_pos_key = None
    for key in ["robot0_eef_pos", "eef_pos"]:
        if key in raw_obs:
            eef_pos_key = key
            break
    if eef_pos_key is None:
        raise RuntimeError(f"Cannot find eef_pos in raw_obs. Available keys: {list(raw_obs.keys())}")
    eef_pos = np.asarray(raw_obs[eef_pos_key], dtype=np.float32).flatten()[:3]

    # End-effector quaternion (4D) -> convert to 6D rotation
    eef_quat_key = None
    for key in ["robot0_eef_quat", "eef_quat"]:
        if key in raw_obs:
            eef_quat_key = key
            break
    if eef_quat_key is None:
        raise RuntimeError(f"Cannot find eef_quat in raw_obs. Available keys: {list(raw_obs.keys())}")
    eef_quat = np.asarray(raw_obs[eef_quat_key], dtype=np.float32).flatten()[:4]
    eef_rot6d = quat_to_rot6d(eef_quat)  # [6,]

    # Gripper state (openness) - typically from gripper joint positions
    gripper_key = None
    for key in ["robot0_gripper_qpos", "gripper_qpos"]:
        if key in raw_obs:
            gripper_key = key
            break

    if gripper_key is not None:
        gripper_qpos = np.asarray(raw_obs[gripper_key], dtype=np.float32).flatten()
        # Take mean of two finger positions as gripper openness (normalized)
        # MimicGen gripper: 0 = closed, 0.04 = open (for Panda)
        gripper_open = np.mean(gripper_qpos)
        # Normalize to roughly [-1, 1] range
        gripper_open = (gripper_open - 0.02) / 0.02
        gripper_open = np.array([gripper_open], dtype=np.float32)
    else:
        # Fallback: use 0 if no gripper state available
        gripper_open = np.array([0.0], dtype=np.float32)

    # Concatenate: [pos(3), rot6d(6), gripper_open(1)] = 10 dims
    gripper_state = np.concatenate([eef_pos, eef_rot6d, gripper_open], axis=-1)
    return gripper_state.astype(np.float32)


# ----------------------------
# PLY-equivalent preprocessing (EXACT MATCH to mimicgen_ply_all_tasks.py)
# ----------------------------

def center_scale_unit_cube_np(pts: np.ndarray) -> np.ndarray:
    """
    Center and isotropically scale to fit in [-1,1]^3.
    Identical to preprocess_mimicgen_voxels.py
    """
    if pts.ndim != 2 or pts.shape[1] not in (3, 6):
        raise RuntimeError(f"pts must be [N,3] or [N,6], got {pts.shape}")
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

def downsample_np(pts: np.ndarray, max_points: int) -> np.ndarray:
    if max_points is None or max_points <= 0:
        return pts
    if pts.shape[0] <= max_points:
        return pts
    idx = np.random.choice(pts.shape[0], max_points, replace=False)
    return pts[idx]

def parse_fixed_bounds_pm(bounds_str: str):
    # expects "xmin,xmax,ymin,ymax,zmin,zmax"
    xmin, xmax, ymin, ymax, zmin, zmax = map(float, bounds_str.split(","))
    pmin = (xmin, ymin, zmin)
    pmax = (xmax, ymax, zmax)
    return (pmin, pmax)

def bounds_dict_to_tuple(bounds_dict):
    """
    Convert a bounds dictionary to (pmin, pmax) tuple format.

    Args:
        bounds_dict: dict with keys xmin, xmax, ymin, ymax, zmin, zmax

    Returns:
        (pmin, pmax) tuple of tuples: ((xmin, ymin, zmin), (xmax, ymax, zmax))
    """
    pmin = (bounds_dict["xmin"], bounds_dict["ymin"], bounds_dict["zmin"])
    pmax = (bounds_dict["xmax"], bounds_dict["ymax"], bounds_dict["zmax"])
    return (pmin, pmax)

# ----------------------------
# Depth conversion (EXACT MATCH to mimicgen_ply_all_tasks.py)
# ----------------------------

def _squeeze_hw(d):
    """Same as squeeze_hw in mimicgen_ply_all_tasks.py"""
    d = np.asarray(d)
    if d.ndim == 3:
        d = np.squeeze(d)
    if d.ndim != 2:
        d = d.reshape(d.shape[-2], d.shape[-1])
    return d

def depth_to_z(depth_raw, mode: str, near_m: float, far_m: float):
    """
    Convert depth buffer to metric depth.

    For LIVE robosuite simulator: use mode="robosuite" (default)
      - Robosuite returns OpenGL depth buffer values ~0.98-0.996
      - Formula: near / (1 - d * (1 - near/far)) = near*far / (far - d*(far-near))

    For OFFLINE HDF5 data: the preprocessing script auto-picks the best mode
      - Often "linear" works for pre-converted depth
    """
    d = _squeeze_hw(depth_raw).astype(np.float32)
    dmax = float(np.nanmax(d)) if np.isfinite(d).any() else 0.0

    if mode == "as_is":
        return d

    if dmax > 1.05:
        # already metric-ish
        return d

    d = np.clip(d, 0.0, 1.0)
    n = float(near_m)
    f = float(far_m)

    if mode == "linear":
        # Linear interpolation (for pre-converted depth)
        return n + d * (f - n)

    if mode == "robosuite" or mode == "opengl_inv_simple":
        # OpenGL depth buffer inversion (matches CU.get_real_depth_map)
        # Formula: near / (1 - d * (1 - near/far))
        # Simplified: near * far / (far - d * (far - near))
        return (n * f) / (f - d * (f - n) + 1e-12)

    if mode == "opengl_ndc":
        z_ndc = 2.0 * d - 1.0
        return (2.0 * n * f) / (f + n - z_ndc * (f - n) + 1e-12)

    raise ValueError(f"unknown depth mode {mode}")

def compute_K_from_fovy(fovy_deg, W, H):
    """
    EXACT COPY from mimicgen_ply_all_tasks.py
    Compute intrinsic matrix from field of view.
    """
    fovy = np.deg2rad(float(fovy_deg))
    fy = (H / 2.0) / np.tan(fovy / 2.0)
    fx = fy * (W / H)
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float32)

def mujoco_near_far_meters(sim):
    """
    EXACT COPY from mimicgen_ply_all_tasks.py
    MuJoCo uses znear/zfar * extent (this is the big gotcha)
    """
    extent = float(sim.model.stat.extent)
    znear = float(sim.model.vis.map.znear) * extent
    zfar  = float(sim.model.vis.map.zfar)  * extent
    return znear, zfar

# ----------------------------
# Token packing (exactly like preprocessing)
# ----------------------------

def pack_tokens_k24_preproc_format(out: dict) -> torch.Tensor:
    """
    EXACT preprocessing format:
      toks = [z(3), z_scale(3), z_depth(1), obj_on(1), z_features(F)]
    Shapes expected: (B,1,24,*)
    """
    if "z" not in out:
        raise RuntimeError("DLP output missing key 'z'")

    z       = out["z"][:, 0]
    z_scale = out["z_scale"][:, 0]
    z_depth = out["z_depth"][:, 0]
    obj_on  = out["obj_on"][:, 0]
    z_feat  = out["z_features"][:, 0]

    if obj_on.ndim == 2:
        obj_on = obj_on.unsqueeze(-1)

    toks = torch.cat([z, z_scale, z_depth, obj_on, z_feat], dim=-1)
    return toks

# ----------------------------
# Helpers
# ----------------------------

def _looks_like_rigid(T: np.ndarray, tol=1e-2) -> bool:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        return False
    if not np.allclose(T[3], [0, 0, 0, 1], atol=tol):
        return False
    R = T[:3, :3]
    cn = np.linalg.norm(R, axis=0)
    if np.any(cn > 3.0) or np.any(cn < 0.2):
        return False
    return True

def _recover_cam2world(T_cam_world_like: np.ndarray, K: np.ndarray) -> np.ndarray:
    T = np.asarray(T_cam_world_like, dtype=np.float64).reshape(4, 4)
    if _looks_like_rigid(T):
        return T.astype(np.float32)

    P = T[:3, :4]
    E = np.linalg.inv(K) @ P
    Tw2c = np.eye(4, dtype=np.float64)
    Tw2c[:3, :4] = E
    Tc2w = np.linalg.inv(Tw2c)
    return Tc2w.astype(np.float32)

# ----------------------------
# Goal Provider for dataset-based init_state + goal pairing
# ----------------------------

class DatasetGoalProvider:
    """
    Provides paired (init_state, goal_tokens) from a preprocessed dataset.
    """

    def __init__(self, data_pkl_path, shuffle=True):
        import pickle

        with open(data_pkl_path, 'rb') as f:
            data = pickle.load(f)

        self.init_states = data['init_states']    # (E, state_dim)
        self.goals = data['goals'][:, 0]          # (E, K, Dtok) - goal per trajectory
        self.first_frame_obs = data['observations'][:, 0]  # (E, K, Dtok) - first frame tokens
        self.path_lengths = data['path_lengths']  # (E,)
        self.num_trajectories = len(self.init_states)

        # Load gripper state if available (E, Tmax, 10)
        self.gripper_state = data.get('gripper_state', None)
        if self.gripper_state is not None:
            print(f"[DatasetGoalProvider] Found gripper_state: shape={self.gripper_state.shape}")
        else:
            print(f"[DatasetGoalProvider] No gripper_state in dataset")

        # Load bg_features if available (E, Tmax, bg_dim)
        self.bg_features = data.get('bg_features', None)
        if self.bg_features is not None:
            print(f"[DatasetGoalProvider] Found bg_features: shape={self.bg_features.shape}")
        else:
            print(f"[DatasetGoalProvider] No bg_features in dataset")

        self.indices = np.arange(self.num_trajectories)
        if shuffle:
            np.random.shuffle(self.indices)

        self.current_idx = 0
        self.last_idx = None

    def get_init_state_and_goal(self):
        """
        Returns the next (init_state, goal_tokens) pair.
        Cycles through all trajectories.
        """
        idx = self.indices[self.current_idx % self.num_trajectories]
        self.current_idx += 1
        self.last_idx = idx

        init_state = self.init_states[idx]  # (state_dim,)
        goal_tokens = self.goals[idx]       # (K, Dtok)

        return init_state, goal_tokens

    def get_goal_gripper_state(self):
        """
        Returns the goal gripper state for the most recently returned trajectory.
        Goal is the last valid timestep's gripper state.

        Returns:
            gripper_state: [10,] array, or None if not available
        """
        if self.gripper_state is None:
            return None
        if self.last_idx is None:
            raise RuntimeError("Call get_init_state_and_goal() first")

        idx = self.last_idx
        path_len = self.path_lengths[idx]
        # Goal gripper state is from the last timestep
        goal_gripper = self.gripper_state[idx, path_len - 1]  # [10,]
        return goal_gripper.astype(np.float32)

    def get_first_frame_gripper_state(self):
        """
        Returns the first frame gripper state for the most recently returned trajectory.
        """
        if self.gripper_state is None:
            return None
        if self.last_idx is None:
            raise RuntimeError("Call get_init_state_and_goal() first")
        return self.gripper_state[self.last_idx, 0].astype(np.float32)  # [10,]

    def get_goal_bg_features(self):
        """
        Returns the goal bg_features for the most recently returned trajectory.
        Goal is the last valid timestep's bg_features.

        Returns:
            bg_features: [bg_dim,] array, or None if not available
        """
        if self.bg_features is None:
            return None
        if self.last_idx is None:
            raise RuntimeError("Call get_init_state_and_goal() first")

        idx = self.last_idx
        path_len = self.path_lengths[idx]
        # Goal bg_features is from the last timestep
        goal_bg = self.bg_features[idx, path_len - 1]  # [bg_dim,]
        return goal_bg.astype(np.float32)

    def get_first_frame_bg_features(self):
        """
        Returns the first frame bg_features for the most recently returned trajectory.
        """
        if self.bg_features is None:
            return None
        if self.last_idx is None:
            raise RuntimeError("Call get_init_state_and_goal() first")
        return self.bg_features[self.last_idx, 0].astype(np.float32)  # [bg_dim,]

    def get_first_frame_tokens(self):
        """
        Returns the preprocessed first frame tokens for the most recently returned trajectory.
        Use this to compare live encoding vs preprocessing.
        """
        if self.last_idx is None:
            raise RuntimeError("Call get_init_state_and_goal() first")
        return self.first_frame_obs[self.last_idx]  # (K, Dtok)

    def reset_sampling(self, shuffle=True):
        """Reset to beginning of trajectories, optionally reshuffle."""
        self.current_idx = 0
        if shuffle:
            np.random.shuffle(self.indices)


# Bounds that were ACTUALLY used during training data voxelization.
# Due to a substring-matching bug in hdf5_states_to_voxels.py, multi-word task
# names (CamelCase env names like "MugCleanup_D0" → "mugcleanup_d0") failed to
# match underscore-separated keys (e.g. "mug_cleanup"), causing them to either:
#   - match a shorter key ("coffee_preparation" matched "coffee")
#   - fall through to per-item bounds (None)
TASK_VOXEL_BOUNDS = {
    # Single-word tasks: matched correctly, used intended task-specific bounds
    "coffee": {"xmin": -0.5, "xmax": 0.4, "ymin": -0.4, "ymax": 0.4, "zmin": 0.2, "zmax": 1.6},
    "kitchen": {"xmin": -1.1, "xmax": 0.3, "ymin": -0.4, "ymax": 0.4, "zmin": 0.7, "zmax": 1.4},
    "square": {"xmin": -0.5, "xmax": 0.4, "ymin": -0.5, "ymax": 0.4, "zmin": 0.0, "zmax": 1.6},
    "stack": {"xmin": -0.7, "xmax": 0.4, "ymin": -0.5, "ymax": 0.5, "zmin": -0.2, "zmax": 1.6},
    "threading": {"xmin": -0.5, "xmax": 0.4, "ymin": -0.4, "ymax": 0.4, "zmin": 0.2, "zmax": 1.6},
    # Multi-word tasks: matched wrong key due to substring bug
    "coffee_preparation": {"xmin": -0.5, "xmax": 0.4, "ymin": -0.4, "ymax": 0.4, "zmin": 0.2, "zmax": 1.6},  # matched "coffee"
    "stack_three": {"xmin": -0.7, "xmax": 0.4, "ymin": -0.5, "ymax": 0.5, "zmin": -0.2, "zmax": 1.6},  # matched "stack"
    # Multi-word tasks: no match, fell through to per-item bounds
    "hammer_cleanup": None,
    "mug_cleanup": None,
    "nut_assembly": None,
    "pick_place": None,
    "three_piece_assembly": None,
}


# ----------------------------
# MimicGenDLPWrapper (EXACT MATCH to preprocessing pipeline)
# ----------------------------

class MimicGenDLPWrapper:
    """
    Online wrapper that EXACTLY matches the preprocessing pipeline:
      mimicgen_ply_all_tasks.py -> preprocess_mimicgen_voxels.py

    Key matching points:
      - Depth conversion: depth_to_z with near*extent, far*extent
      - Camera transform: flip y/z in cam coords, then R @ pts + pos
      - Intrinsics: compute_K_from_fovy
      - Normalization: center_scale_unit_cube (enabled by default)
      - Voxelization: VoxelGridXYZ with per_item bounds
    """

    def __init__(
        self,
        env,
        dlp_model,
        device,
        cams=("agentview", "sideview"),

        # preprocessing-equivalent knobs
        max_points=20000,
        normalize_to_unit_cube=True,  # DEFAULT TRUE to match preprocess_mimicgen_voxels.py
        include_rgb=True,

        # depth conversion mode
        # "robosuite" (default): correct for LIVE simulator (OpenGL depth buffer inversion)
        # "linear": for offline HDF5 data where preprocessing auto-picked this mode
        depth_mode="robosuite",

        # voxelization knobs — must match DLP training resolution
        grid_dhw=(128, 128, 128),
        voxel_mode="avg_rgb",
        bounds_mode="per_item",  # DEPRECATED: now always uses task-specific bounds
        fixed_bounds="-1,1,-1,1,-1,1",  # DEPRECATED: now always uses task-specific bounds
        global_bounds_pm=None,  # DEPRECATED: now always uses task-specific bounds

        # calibration
        calib_h5_path=None,
        use_h5_calib=False,
        crop_bounds={"xmin": -1.1, "xmax": 0.9, "ymin": -0.5, "ymax": 0.5, "zmin": -0.2, "zmax": 2.5},

        # goal
        get_goal_raw_obs_fn=None,
        goal_provider=None,
        random_init=False,  # If True, use random env reset instead of dataset init states
        success_key_candidates=("success", "task_success", "is_success"),
        pixel_stride=1,  # DEFAULT 1 to match mimicgen_ply_all_tasks.py
        max_points_backproject=200_000,  # matches mimicgen_ply_all_tasks.py --max-points

        # voxel saving for offline analysis
        save_voxels_dir=None,
        save_points=False,     # also save raw point clouds

        # task-specific bounds
        task=None,  # Task name (e.g., "threading", "hammer_cleanup") - REQUIRED for task-specific bounds
    ):
        self.env = env
        self.dlp = dlp_model
        self.device = torch.device(device)
        self.cams = list(cams)

        self.max_points = int(max_points)
        self.normalize_to_unit_cube = bool(normalize_to_unit_cube)
        self.include_rgb = bool(include_rgb)
        self.depth_mode = str(depth_mode)

        self.grid_dhw = tuple(grid_dhw)
        self.voxel_mode = str(voxel_mode)
        self.bounds_mode = str(bounds_mode)
        self.pixel_stride = int(pixel_stride)
        self.max_points_backproject = int(max_points_backproject)

        self.get_goal_raw_obs_fn = get_goal_raw_obs_fn
        self.goal_provider = goal_provider
        self.random_init = bool(random_init)
        self.success_key_candidates = tuple(success_key_candidates)

        self.last_raw_obs = None
        self.last_info = None
        self.goal_vec = None
        self.last_toks = None
        self.decoded_vox = None
        self.last_vox = None
        self.vox = None
        self.last_gripper_state = None  # [10,] gripper state from last obs
        self.goal_gripper_state = None  # [10,] gripper state for goal
        self.last_bg_features = None  # [bg_dim,] background features from last obs
        self.goal_bg_features = None  # [bg_dim,] background features for goal

        # --- load calib ---
        self.calib = {}
        self.crop_bounds = crop_bounds
        self.use_h5_calib = bool(use_h5_calib)
        self.calib_h5_path = calib_h5_path
        self._h5_calib = None

        if self.use_h5_calib and self.calib_h5_path is not None:
            self._load_h5_calib(self.calib_h5_path)

        # --- voxel bounds (must match what was used during training data generation) ---
        self.task = task
        if task is None:
            raise RuntimeError(
                "task parameter is REQUIRED for MimicGenDLPWrapper. "
                "Pass task name (e.g., 'threading', 'mug_cleanup') to select correct voxel bounds."
            )
        if task not in TASK_VOXEL_BOUNDS:
            raise RuntimeError(
                f"Unknown task '{task}'. Available tasks: {list(TASK_VOXEL_BOUNDS.keys())}"
            )

        bounds_entry = TASK_VOXEL_BOUNDS[task]
        if bounds_entry is not None:
            self.bounds_pm = bounds_dict_to_tuple(bounds_entry)
            print(f"[MimicGenDLPWrapper] Using fixed bounds for '{task}': {self.bounds_pm}")
        else:
            self.bounds_pm = None  # per-item: VoxelGridXYZ computes from each frame's data
            print(f"[MimicGenDLPWrapper] Using per-item bounds for '{task}' (matching training)")

        # precompute pixel grids cache
        self._precomputed = {}

        # voxel saving
        self.save_voxels_dir = save_voxels_dir
        self.save_points = save_points
        self._save_counter = 0
        self._save_ep_counter = 0
        if self.save_voxels_dir is not None:
            import os
            os.makedirs(self.save_voxels_dir, exist_ok=True)
            print(f"[MimicGenDLPWrapper] Saving voxels to: {self.save_voxels_dir}")

    def _get_robosuite_env_and_sim(self):
        """Unwrap robomimic EnvRobosuite to reach a robosuite env with .sim."""
        e = self.env
        seen = set()

        for _ in range(10):
            if id(e) in seen:
                break
            seen.add(id(e))

            if hasattr(e, "sim"):
                return e, e.sim

            for attr in ("env", "_env", "base_env", "unwrapped"):
                if hasattr(e, attr):
                    e2 = getattr(e, attr)
                    if e2 is not None and e2 is not e:
                        e = e2
                        break
            else:
                break

        raise AttributeError(
            f"Could not find underlying robosuite env with `.sim`. "
            f"Got wrapper chain ending at type={type(e)} with attrs={dir(e)}"
        )

    def _ensure_env_calib(self, raw_obs):
        """
        Build calibration EXACTLY like mimicgen_ply_all_tasks.py:
          - K from fovy
          - near/far * extent
          - R from cam_xmat (NO transpose)
          - pos from cam_xpos
        """
        _, sim = self._get_robosuite_env_and_sim()

        # Get near/far in meters (multiply by extent!)
        near_m, far_m = mujoco_near_far_meters(sim)

        for cam in self.cams:
            depth = np.asarray(raw_obs[f"{cam}_depth"])
            # Handle various depth formats: (1, H, W), (H, W, 1), or (H, W)
            if depth.ndim == 3 and depth.shape[0] == 1:
                depth = depth[0]  # (1, H, W) -> (H, W)
            elif depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]  # (H, W, 1) -> (H, W)
            depth = np.squeeze(depth)  # Remove any remaining singleton dims
            H, W = int(depth.shape[0]), int(depth.shape[1])

            prev = self.calib.get(cam, None)
            if prev is not None and prev["H"] == H and prev["W"] == W:
                continue

            # Use H5 calibration if available
            # if self._h5_calib is not None and cam in self._h5_calib:
            #     h5_cam = self._h5_calib[cam]
            #     K = h5_cam["K"]
            #     # For H5 calib, we still use the standard transform
            #     cam_id = sim.model.camera_name2id(cam)
            #     pos = np.array(sim.data.cam_xpos[cam_id], dtype=np.float32)
            #     R = np.array(sim.data.cam_xmat[cam_id], dtype=np.float32).reshape(3, 3)
            #     self.calib[cam] = {"K": K, "R": R, "pos": pos, "near_m": near_m, "far_m": far_m, "H": H, "W": W}
            #     continue

            # EXACT MATCH to mimicgen_ply_all_tasks.py
            # Intrinsics from fovy
            cam_id = sim.model.camera_name2id(cam)
            fovy = float(sim.model.cam_fovy[cam_id])
            K = compute_K_from_fovy(fovy, W=W, H=H)

            # Extrinsics: pos and R (NO transpose, NO axis flip - we do that in transform)
            pos = np.array(sim.data.cam_xpos[cam_id], dtype=np.float32)
            R = np.array(sim.data.cam_xmat[cam_id], dtype=np.float32).reshape(3, 3)

            self.calib[cam] = {"K": K, "R": R, "pos": pos, "near_m": near_m, "far_m": far_m, "H": H, "W": W}

    def crop_world(self, xyz, rgb=None):
        """EXACT MATCH to maybe_crop in mimicgen_ply_all_tasks.py"""
        b = self.crop_bounds
        if b is None:
            return (xyz, rgb) if rgb is not None else xyz

        m = (
            (xyz[:, 0] >= b["xmin"]) & (xyz[:, 0] <= b["xmax"]) &
            (xyz[:, 1] >= b["ymin"]) & (xyz[:, 1] <= b["ymax"]) &
            (xyz[:, 2] >= b["zmin"]) & (xyz[:, 2] <= b["zmax"])
        )
        if not np.any(m):
            return (xyz, rgb) if rgb is not None else xyz
        xyz2 = xyz[m]
        if rgb is None:
            return xyz2
        return xyz2, rgb[m]

    def _load_h5_calib(self, h5_path):
        """Load calibration from HDF5 for comparison/fallback."""
        self._h5_calib = {}
        with h5py.File(h5_path, "r") as h5:
            for cam in self.cams:
                if f"meta/cameras/{cam}" not in h5:
                    continue
                g = h5[f"meta/cameras/{cam}"]
                K = np.asarray(g["K"], dtype=np.float32)
                T_raw = np.asarray(g["T_cam_world"], dtype=np.float32)

                near = g.attrs.get("near", None)
                far = g.attrs.get("far", None)
                if near is None and "near" in g:
                    near = float(np.asarray(g["near"]))
                if far is None and "far" in g:
                    far = float(np.asarray(g["far"]))

                Tc2w = _recover_cam2world(T_raw, K.astype(np.float64))

                self._h5_calib[cam] = {
                    "K": K,
                    "Tc2w": Tc2w,
                    "T_raw": T_raw,
                    "near": near,
                    "far": far,
                }

    def _get_rgbd_from_obs(self, obs, cam):
        rgb_key = f"{cam}_image"
        dep_key = f"{cam}_depth"
        if rgb_key not in obs:
            raise KeyError(f"obs missing key {rgb_key}")
        if dep_key not in obs:
            raise KeyError(f"obs missing key {dep_key}")
        return np.asarray(obs[rgb_key]), np.asarray(obs[dep_key])

    def _backproject_cam(self, cam, depth_raw):
        """
        EXACT MATCH to backproject() in mimicgen_ply_all_tasks.py
        Returns pts in CAMERA coordinates (OpenCV convention: x-right, y-down, z-forward)
        """
        calib = self.calib[cam]
        K = calib["K"]
        near_m = calib["near_m"]
        far_m = calib["far_m"]

        # Convert depth using the same function as preprocessing
        z = depth_to_z(depth_raw, mode=self.depth_mode, near_m=near_m, far_m=far_m)
        H, W = z.shape

        # Pixel grid with stride (same as preprocessing)
        vv = np.arange(0, H, self.pixel_stride, dtype=np.int32)
        uu = np.arange(0, W, self.pixel_stride, dtype=np.int32)
        U, V = np.meshgrid(uu, vv)

        Z = z[V, U]
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

    def _cam_to_world(self, pts_cam, cam):
        """
        EXACT MATCH to build_points_for_cam() transform in mimicgen_ply_all_tasks.py:
          # --- CV -> MuJoCo(OpenGL) camera coords ---
          pts_gl = pts_cam.copy()
          pts_gl[:, 1] *= -1.0   # y: down -> up
          pts_gl[:, 2] *= -1.0   # z: forward -> backward
          # --- camera -> world ---
          pts_world = (R @ pts_gl.T).T + pos
        """
        calib = self.calib[cam]
        R = calib["R"]
        pos = calib["pos"]

        # CV -> MuJoCo(OpenGL) camera coords
        pts_gl = pts_cam.copy()
        pts_gl[:, 1] *= -1.0   # y: down -> up
        pts_gl[:, 2] *= -1.0   # z: forward -> backward

        # camera -> world
        pts_world = (R @ pts_gl.T).T + pos

        return pts_world.astype(np.float32)

    def _fuse_xyzrgb_world(self, raw_obs) -> np.ndarray:
        """
        Fuse point clouds from all cameras.
        EXACT MATCH to the loop in process_task() in mimicgen_ply_all_tasks.py
        """
        all_xyz, all_rgb = [], []

        for cam in self.cams:
            rgb, depth = self._get_rgbd_from_obs(raw_obs, cam)
            pts_cam, idxs = self._backproject_cam(cam, depth)
            if pts_cam.shape[0] == 0:
                continue

            v = idxs[:, 0]
            u = idxs[:, 1]

            # Handle RGB format
            if rgb.ndim == 4 and rgb.shape[0] == 1:
                rgb = rgb[0]
            if rgb.ndim == 3 and rgb.shape[0] in (1, 3, 4) and rgb.shape[-1] not in (1, 3, 4):
                rgb = np.transpose(rgb, (1, 2, 0))
            if rgb.ndim != 3 or rgb.shape[2] < 3:
                raise RuntimeError(f"RGB for cam={cam} must be HWC with >=3 channels, got {rgb.shape}")

            rgb_pts = rgb[v, u, :3].astype(np.float32)
            if rgb_pts.max() > 1.5:
                rgb_pts /= 255.0

            # Transform to world (EXACT MATCH to preprocessing)
            pts_world = self._cam_to_world(pts_cam, cam)

            # Crop (EXACT MATCH to preprocessing)
            pts_world, rgb_pts = self.crop_world(pts_world, rgb_pts)

            if pts_world.shape[0] == 0:
                continue

            all_xyz.append(pts_world)
            all_rgb.append(rgb_pts)

        if len(all_xyz) == 0:
            raise RuntimeError("No points reconstructed from any camera (all empty).")

        xyz = np.concatenate(all_xyz, axis=0)
        rgb = np.concatenate(all_rgb, axis=0)

        # Remove non-finite points
        if not np.isfinite(xyz).all():
            finite = np.isfinite(xyz).all(axis=1)
            xyz = xyz[finite]
            rgb = rgb[finite]
        if xyz.shape[0] == 0:
            raise RuntimeError("All reconstructed points were non-finite.")

        # Max points cap (same as --max-points in preprocessing)
        if self.max_points_backproject > 0 and xyz.shape[0] > self.max_points_backproject:
            sel = np.random.choice(xyz.shape[0], self.max_points_backproject, replace=False)
            xyz, rgb = xyz[sel], rgb[sel]

        if self.include_rgb:
            return np.concatenate([xyz, rgb], axis=-1).astype(np.float32)
        return xyz.astype(np.float32)

    def _preprocess_points_like_TODataset(self, pts: np.ndarray) -> torch.Tensor:
        """
        EXACT MATCH to preprocess_mimicgen_voxels.py:
          - center_scale_unit_cube (if enabled)
          - downsample to max_points
        """
        if self.normalize_to_unit_cube:
            pts = center_scale_unit_cube_np(pts)

        pts = downsample_np(pts, self.max_points)
        return torch.from_numpy(pts).float()

    def _save_voxel(self, vox: torch.Tensor, pts_t: torch.Tensor = None):
        """Save voxel (and optionally points) to disk."""
        import os

        ep_dir = os.path.join(self.save_voxels_dir, f"ep_{self._save_ep_counter:04d}")
        os.makedirs(ep_dir, exist_ok=True)

        # Save voxel
        vox_path = os.path.join(ep_dir, f"frame_{self._save_counter:06d}_voxels.pt")
        torch.save(vox.detach().cpu(), vox_path)

        # Save points if requested
        if pts_t is not None:
            pts_path = os.path.join(ep_dir, f"frame_{self._save_counter:06d}_points.pt")
            torch.save(pts_t.detach().cpu(), pts_path)

        self._save_counter += 1

    def _voxelize_via_wrapper(self, pts_t: torch.Tensor) -> torch.Tensor:
        """
        Voxelize points using bounds from TASK_VOXEL_BOUNDS.
        Uses VoxelGridXYZ with bounds=(pmin, pmax) or None for per-item.
        """
        D, H, W = map(int, self.grid_dhw)

        if pts_t.ndim != 2 or pts_t.shape[1] not in (3, 6):
            raise RuntimeError(f"pts_t must be [N,3] or [N,6], got {tuple(pts_t.shape)}")

        pts_t = pts_t.to(self.device)

        xyz = pts_t[:, :3]
        colors = None
        # rgb_occ / rgbo modes need avg_rgb voxelization + appended occupancy channel
        effective_mode = self.voxel_mode
        if self.voxel_mode in ("rgb_occ", "rgbo"):
            effective_mode = "avg_rgb"

        if effective_mode == "avg_rgb":
            if pts_t.shape[1] != 6:
                raise RuntimeError("voxel_mode='avg_rgb' requires include_rgb=True and points [N,6].")
            colors = pts_t[:, 3:6]

        vg = VoxelGridXYZ(
            points_xyz=xyz,
            colors=colors,
            grid_whd=(W, H, D),
            bounds=self.bounds_pm,
            mode=effective_mode,
        )
        vox = vg.to_dense()  # [3, D, H, W] for avg_rgb

        # Append occupancy channel for rgb_occ/rgbo (matches rgb_to_rgbo_inplace.py)
        if self.voxel_mode in ("rgb_occ", "rgbo"):
            occ = (vox.abs().sum(dim=0, keepdim=True) > 0).float()
            vox = torch.cat([vox, occ], dim=0)  # [4, D, H, W]

        return vox

    @torch.no_grad()
    def encode_tokens(self, raw_obs):
        self._ensure_env_calib(raw_obs)
        pts = self._fuse_xyzrgb_world(raw_obs)

        pts_t = self._preprocess_points_like_TODataset(pts)

        vox = self._voxelize_via_wrapper(pts_t)

        # Save voxels if enabled
        if self.save_voxels_dir is not None:
            self._save_voxel(vox, pts_t if self.save_points else None)

        vox_b = vox.unsqueeze(0)
        out = self.dlp(vox_b, deterministic=True, warmup=False, with_loss=False)
        z_bg = out["z_bg_features"]

        toks = pack_tokens_k24_preproc_format(out)
        toks_np = toks[0].detach().cpu().numpy().astype(np.float32)
        flat = toks_np.reshape(-1).astype(np.float32)

        # Store for comparison
        self.last_toks = toks_np
        self.decoded_vox = None
        self.last_vox = vox.detach().cpu().numpy()
        self.vox = None  # don't hold GPU tensor

        # Store background features: z_bg is [B, 1, bg_dim], extract [bg_dim]
        self.last_bg_features = z_bg[0, 0].detach().cpu().numpy().astype(np.float32)

        # Free DLP intermediates before diffusion model runs
        del out, vox_b, vox
        torch.cuda.empty_cache()

        return flat, toks_np, self.last_vox

    def _get_success_from_info(self, info):
        if info is None:
            return None
        for k in self.success_key_candidates:
            if k in info:
                v = info[k]
                # Handle nested dict like {'task': True} from MimicGen
                if isinstance(v, dict):
                    # Check for 'task' key or take any True value
                    if 'task' in v:
                        return bool(v['task'])
                    # Fallback: return True if any value is True
                    return any(bool(val) for val in v.values())
                if isinstance(v, (bool, np.bool_)):
                    return bool(v)
                if isinstance(v, (int, float, np.integer, np.floating)):
                    return bool(v)
        return None

    def unpack_tokens_to_dlp_format(self, toks_np):
        """
        Unpack flat tokens (K, Dtok) back to DLP dict format for decoding.
        Inverse of pack_tokens_k24_preproc_format.

        Token format: [z(3), z_scale(3), z_depth(1), obj_on(1), z_features(F)]
        """
        toks = torch.from_numpy(toks_np).float().to(self.device)
        if toks.ndim == 2:
            toks = toks.unsqueeze(0)  # (1, K, Dtok)

        # Add the extra dimension that DLP expects: (B, 1, K, *)
        toks = toks.unsqueeze(1)  # (B, 1, K, Dtok)

        z = toks[..., :3]           # (B, 1, K, 3)
        z_scale = toks[..., 3:6]    # (B, 1, K, 3)
        z_depth = toks[..., 6:7]    # (B, 1, K, 1)
        obj_on = toks[..., 7:8]     # (B, 1, K, 1)
        z_features = toks[..., 8:]  # (B, 1, K, F)

        return {
            "z": z,
            "z_scale": z_scale,
            "z_depth": z_depth,
            "obj_on": obj_on,
            "z_features": z_features,
        }

    @torch.no_grad()
    def decode_particles(self, toks_np):
        """
        Decode particle tokens through DLP decoder to get reconstructed voxels.

        Args:
            toks_np: (K, Dtok) numpy array of packed tokens

        Returns:
            dict with:
                - fg_only: (3, D, H, W) torch tensor - foreground objects only
                - rec_rgb: (3, D, H, W) torch tensor - full reconstruction
        """
        dlp_dict = self.unpack_tokens_to_dlp_format(toks_np)

        z = dlp_dict["z"]
        z_scale = dlp_dict["z_scale"]
        z_depth = dlp_dict["z_depth"]
        obj_on = dlp_dict["obj_on"]
        z_features = dlp_dict["z_features"]

        dec = self.dlp.decode_all(
            z, z_scale, z_features, obj_on, z_depth,
            None, None,
            warmup=False
        )

        if not isinstance(dec, dict):
            raise TypeError(f"decode_all returned {type(dec)}; expected dict")

        rec_rgb = dec["rec_rgb"]
        fg_only = dec["dec_objects_trans"]

        return {
            "fg_only": fg_only[0],
            "rec_rgb": rec_rgb[0],
        }

    @torch.no_grad()
    def compare_live_vs_dataset_particles(self, dataset_tokens=None, log_to_wandb=False, step=None, prefix="particle_comparison"):
        """
        Compare live-encoded particles vs dataset particles by decoding both
        through the DLP decoder and comparing reconstructed voxels.
        """
        if self.last_toks is None or self.last_vox is None:
            raise RuntimeError("No live encoding available. Call reset() or step() first.")

        if dataset_tokens is None:
            if self.goal_provider is None:
                raise RuntimeError("No dataset_tokens provided and no goal_provider available")
            dataset_tokens = self.goal_provider.get_first_frame_tokens()

        dataset_tokens = np.asarray(dataset_tokens, dtype=np.float32)
        live_tokens = self.last_toks
        gt_vox = self.last_vox

        decoded_dataset = self.decode_particles(dataset_tokens)
        decoded_live = self.decode_particles(live_tokens)

        dataset_fg = decoded_dataset["fg_only"]
        live_fg = decoded_live["fg_only"]
        dataset_rec = decoded_dataset["rec_rgb"]
        live_rec = decoded_live["rec_rgb"]

        token_l2 = np.linalg.norm(live_tokens - dataset_tokens, axis=-1)

        dataset_fg_np = dataset_fg.detach().cpu().numpy()
        live_fg_np = live_fg.detach().cpu().numpy()

        vox_mse_live_vs_dataset = float(np.mean((live_fg_np - dataset_fg_np) ** 2))

        result = {
            "token_l2_mean": float(token_l2.mean()),
            "token_l2_max": float(token_l2.max()),
            "vox_mse_live_vs_dataset": vox_mse_live_vs_dataset,
            "gt_vox": gt_vox,
            "decoded_dataset_fg": dataset_fg,
            "decoded_dataset_rec": dataset_rec,
            "decoded_live_fg": live_fg,
            "decoded_live_rec": live_rec,
            "dataset_tokens": dataset_tokens,
            "live_tokens": live_tokens,
        }

        if log_to_wandb:
            try:
                import wandb

                wandb.log({
                    f"{prefix}/token_l2_mean": result["token_l2_mean"],
                    f"{prefix}/token_l2_max": result["token_l2_max"],
                    f"{prefix}/vox_mse_live_vs_dataset": result["vox_mse_live_vs_dataset"],
                }, step=step)

                log_rgb_voxels(
                    name=f"{prefix}/gt_vox_from_sim",
                    rgb_vol=gt_vox,
                    alpha_vol=None, KPx=None, step=step,
                    mode="splat", topk=60000, alpha_thresh=0.05, pad=2.0, show_axes=True,
                )

                log_rgb_voxels(
                    name=f"{prefix}/decoded_dataset_particles_fg",
                    rgb_vol=dataset_fg,
                    alpha_vol=None, KPx=None, step=step,
                    mode="splat", topk=60000, alpha_thresh=0.05, pad=2.0, show_axes=True,
                )

                log_rgb_voxels(
                    name=f"{prefix}/decoded_live_particles_fg",
                    rgb_vol=live_fg,
                    alpha_vol=None, KPx=None, step=step,
                    mode="splat", topk=60000, alpha_thresh=0.05, pad=2.0, show_axes=True,
                )

                log_rgb_voxels(
                    name=f"{prefix}/decoded_dataset_particles_rec",
                    rgb_vol=dataset_rec,
                    alpha_vol=None, KPx=None, step=step,
                    mode="splat", topk=60000, alpha_thresh=0.05, pad=2.0, show_axes=True,
                )

                log_rgb_voxels(
                    name=f"{prefix}/decoded_live_particles_rec",
                    rgb_vol=live_rec,
                    alpha_vol=None, KPx=None, step=step,
                    mode="splat", topk=60000, alpha_thresh=0.05, pad=2.0, show_axes=True,
                )

            except ImportError:
                pass
            except Exception as e:
                print(f"[WARNING] Failed to log to wandb: {e}")

        return result

    def reset(self, **kwargs):
        """Reset the environment and set up goal conditioning.

        If random_init=True (set in __init__), uses random environment reset.
        Otherwise, uses dataset init states from goal_provider.
        Goal tokens always come from dataset when goal_provider is available.
        """
        # Increment episode counter for voxel saving
        if self.save_voxels_dir is not None:
            self._save_ep_counter += 1
            self._save_counter = 0

        if self.goal_provider is not None:
            init_state, goal_tokens = self.goal_provider.get_init_state_and_goal()

            if self.random_init:
                # Random initialization - ignore dataset init_state
                raw_obs = self.env.reset(**kwargs)
            else:
                # Dataset initialization - use exact init_state from dataset
                raw_obs = self.env.reset_to({"states": init_state})

            self.last_raw_obs = raw_obs
            self._ensure_env_calib(raw_obs)
            self.last_info = None

            obs_vec, _, _ = self.encode_tokens(raw_obs)

            # Extract gripper state from live observation
            self.last_gripper_state = extract_gripper_state_from_obs(raw_obs)

            # Goal tokens always come from dataset
            self.goal_vec = goal_tokens.reshape(-1).astype(np.float32)

            # Goal gripper state from dataset if available, else use current
            self.goal_gripper_state = self.goal_provider.get_goal_gripper_state()
            if self.goal_gripper_state is None:
                self.goal_gripper_state = self.last_gripper_state.copy()

            # Goal bg_features from dataset if available, else use current
            self.goal_bg_features = self.goal_provider.get_goal_bg_features()
            if self.goal_bg_features is None:
                self.goal_bg_features = self.last_bg_features.copy() if self.last_bg_features is not None else None

            return obs_vec

        # Legacy path: standard reset
        raw_obs = self.env.reset(**kwargs)
        self.last_raw_obs = raw_obs
        self._ensure_env_calib(raw_obs)
        self.last_info = None

        obs_vec, _, _ = self.encode_tokens(raw_obs)

        # Extract gripper state from live observation
        self.last_gripper_state = extract_gripper_state_from_obs(raw_obs)

        if self.get_goal_raw_obs_fn is not None:
            goal_raw = self.get_goal_raw_obs_fn(self.env, raw_obs)
            goal_vec, _, _ = self.encode_tokens(goal_raw)
            self.goal_vec = goal_vec
            # Extract goal gripper state
            self.goal_gripper_state = extract_gripper_state_from_obs(goal_raw)
            # Extract goal bg_features (already stored by encode_tokens)
            self.goal_bg_features = self.last_bg_features.copy() if self.last_bg_features is not None else None
        else:
            self.goal_vec = obs_vec.copy()
            self.goal_gripper_state = self.last_gripper_state.copy()
            self.goal_bg_features = self.last_bg_features.copy() if self.last_bg_features is not None else None

        return obs_vec

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        self.last_raw_obs = raw_obs
        self.last_info = info

        obs_vec, _, _ = self.encode_tokens(raw_obs)

        # Extract gripper state from live observation
        self.last_gripper_state = extract_gripper_state_from_obs(raw_obs)

        success = self._get_success_from_info(info)
        if success is not None:
            info = dict(info)
            info["success"] = success
        return obs_vec, float(reward), bool(done), info

    def make_cond(self, obs_vec, horizon):
        """
        Create condition dict for diffusion model.

        Returns:
            cond_np: dict mapping timestep -> observations (without gripper state)
        """
        if self.goal_vec is None:
            raise RuntimeError("Call reset() before make_cond().")
        return {
            0: obs_vec.copy(),
            horizon - 1: self.goal_vec.copy(),
        }

    def get_gripper_cond(self, horizon):
        """
        Get gripper state conditions for current and goal.

        Returns:
            dict mapping timestep -> gripper_state [10,]
            Returns None if gripper state not available.
        """
        if self.last_gripper_state is None or self.goal_gripper_state is None:
            return None
        return {
            0: self.last_gripper_state.copy(),
            horizon - 1: self.goal_gripper_state.copy(),
        }

    def get_bg_cond(self, horizon):
        """
        Get background features conditions for current and goal.

        Returns:
            dict mapping timestep -> bg_features [bg_dim,]
            Returns None if bg_features not available.
        """
        if self.last_bg_features is None or self.goal_bg_features is None:
            return None
        return {
            0: self.last_bg_features.copy(),
            horizon - 1: self.goal_bg_features.copy(),
        }
