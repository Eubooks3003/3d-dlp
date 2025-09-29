"""
Open3D GUI For live PLY viewing
"""

# utils/o3d_viewer.py
import threading
import time
import numpy as np
import open3d as o3d

def _build_point_cloud(rgb_hw3, depth_hw, intr, *, depth_is_normalized=False, near=None, far=None,
                       sample_step=1, max_points=None):
    H, W = depth_hw.shape
    fx, fy, cx, cy = float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])

    d = depth_hw.astype(np.float32)
    if depth_is_normalized:
        assert near is not None and far is not None
        d = near + d * (far - near)
        d[d >= (far - 1e-6)] = np.inf

    valid = np.isfinite(d) & (d > 0)

    # optional stride / cap
    stride = max(1, int(sample_step))
    ys, xs = np.where(valid)
    if max_points is not None and ys.size > max_points:
        # approximate uniform thinning
        stride = max(stride, int(np.ceil(np.sqrt(ys.size / float(max_points)))))
    if stride > 1:
        ys = ys[::stride]; xs = xs[::stride]

    z = d[ys, xs]
    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)

    if rgb_hw3 is not None:
        rgb = np.clip(rgb_hw3, 0.0, 1.0)
        cols = rgb[ys, xs, :].reshape(-1, 3).astype(np.float32)
    else:
        cols = np.full((xyz.shape[0], 3), 0.5, dtype=np.float32)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(cols)
    return pc

class O3DPointCloudViewer:
    """
    A non-blocking Open3D viewer you can start once and update with new RGBD.
    - Call start() once (creates a window and a render loop thread).
    - Call set_data(...) whenever you have new rgb/depth to show.
    - Call close() when done.
    """
    def __init__(self, title="RGBD Point Cloud", width=960, height=720):
        self.title = title
        self.width = width
        self.height = height
        self.vis = None
        self.geom = None
        self._lock = threading.Lock()
        self._pending_pc = None
        self._running = False
        self._thread = None

    def start(self):
        if self._running:  # already started
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=self.title, width=self.width, height=self.height, visible=True)
        render_opt = self.vis.get_render_option()
        render_opt.background_color = np.array([0.02, 0.02, 0.02])

        while self._running and self.vis is not None:
            # install / update geometry if new data arrives
            with self._lock:
                pc = self._pending_pc
                self._pending_pc = None

            if pc is not None:
                if self.geom is None:
                    self.geom = pc
                    self.vis.add_geometry(self.geom)
                else:
                    self.geom.points = pc.points
                    self.geom.colors = pc.colors
                    self.geom.normals = o3d.utility.Vector3dVector()  # clear normals (if any)
                    self.vis.update_geometry(self.geom)

            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.01)

        # cleanup
        try:
            if self.vis is not None:
                self.vis.destroy_window()
        except Exception:
            pass
        self.vis = None
        self.geom = None

    def set_data(self, *, rgb_hw3, depth_hw, intr, depth_is_normalized=False, near=None, far=None,
                 sample_step=1, max_points=250_000):
        pc = _build_point_cloud(
            rgb_hw3, depth_hw, intr,
            depth_is_normalized=depth_is_normalized, near=near, far=far,
            sample_step=sample_step, max_points=max_points
        )
        with self._lock:
            self._pending_pc = pc

    def close(self):
        self._running = False
