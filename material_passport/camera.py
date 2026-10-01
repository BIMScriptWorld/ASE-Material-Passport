"""Camera model + ground-truth trajectory, with world->pixel projection."""
from __future__ import annotations

import os

import numpy as np
from projectaria_tools.projects import ase


class SceneCamera:
    """ASE RGB fisheye camera + per-frame ground-truth poses for one scene."""

    def __init__(self, scene_dir: str):
        self.calib = ase.get_ase_rgb_calibration()
        self.width, self.height = (int(v) for v in self.calib.get_image_size())
        self.valid_radius = float(self.calib.get_valid_radius())
        self.principal_point = np.asarray(self.calib.get_principal_point(),
                                          dtype=float)

        traj = ase.readers.read_trajectory_file(
            os.path.join(scene_dir, "trajectory.csv"))
        self.timestamps = traj["timestamps"]
        # Precompute T_world_camera (4x4) for every frame.
        T_dev_cam = np.asarray(
            self.calib.get_transform_device_camera().to_matrix(), dtype=float)
        self.T_world_camera = [
            np.asarray(T.to_matrix(), dtype=float) @ T_dev_cam
            for T in traj["Ts_world_from_device"]
        ]
        self.T_camera_world = [np.linalg.inv(T) for T in self.T_world_camera]
        self.num_frames = len(self.T_world_camera)

    def project(self, points_world: np.ndarray, frame_idx: int):
        """Project world points into the given frame.

        Returns ``(uv, ray_dist_m, valid_index)`` where ``uv`` is (M, 2) pixel
        coordinates, ``ray_dist_m`` is (M,) distance from camera center along
        the ray (metres), and ``valid_index`` maps each returned row back to the
        input row. Points behind the camera or outside the valid image circle
        are dropped.
        """
        T_cw = self.T_camera_world[frame_idx]
        n = points_world.shape[0]
        homog = np.concatenate([points_world, np.ones((n, 1))], axis=1)
        cam = (homog @ T_cw.T)[:, :3]

        uv = np.empty((n, 2), dtype=float)
        dist = np.empty(n, dtype=float)
        ok = np.zeros(n, dtype=bool)
        cx, cy = self.principal_point
        r2 = self.valid_radius ** 2
        for i in range(n):
            p = cam[i]
            if p[2] <= 0.0:
                continue
            px = self.calib.project(p)
            if px is None:
                continue
            du, dv = px[0] - cx, px[1] - cy
            if du * du + dv * dv > r2:
                continue
            if 0.0 <= px[0] < self.width and 0.0 <= px[1] < self.height:
                uv[i] = px
                dist[i] = float(np.linalg.norm(p))
                ok[i] = True
        idx = np.nonzero(ok)[0]
        return uv[idx], dist[idx], idx
