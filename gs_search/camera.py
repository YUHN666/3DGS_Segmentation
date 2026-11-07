from dataclasses import dataclass
import numpy as np


@dataclass
class Camera:
    c2w: np.ndarray
    K: np.ndarray
    width: int
    height: int

    def __post_init__(self):
        self.c2w = np.array(self.c2w, dtype=np.float64)
        self.K = np.array(self.K, dtype=np.float64)
        if self.c2w.shape != (4, 4) or self.K.shape != (3, 3):
            raise ValueError("Expected 4x4 camera_to_world and 3x3 intrinsics")
        if not np.isfinite(self.c2w).all() or not np.isfinite(self.K).all():
            raise ValueError("Non-finite camera")
        R = self.c2w[:3, :3]
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-4) or np.linalg.det(R) < .999:
            raise ValueError("Camera rotation must be right-handed and orthonormal")
        if not np.allclose(self.c2w[3], [0, 0, 0, 1]):
            raise ValueError("Invalid homogeneous pose")
        if min(self.width, self.height, self.K[0, 0], self.K[1, 1]) <= 0:
            raise ValueError("Invalid camera resolution/focal length")

    @property
    def position(self):
        return self.c2w[:3, 3].copy()

    def moved(self, position):
        pose = self.c2w.copy()
        pose[:3, 3] = position
        return Camera(pose, self.K.copy(), self.width, self.height)

    def as_dict(self):
        return dict(camera_to_world=self.c2w.tolist(), K=self.K.tolist(),
                    width=self.width, height=self.height, convention="OpenCV +X right +Y down +Z forward")

    @classmethod
    def from_dict(cls, d):
        return cls(d["camera_to_world"], d["K"], d["width"], d["height"])


def look_at(position, direction, up, K, width, height):
    f = np.asarray(direction, float)
    if np.linalg.norm(f) < 1e-9:
        raise ValueError("Zero look direction")
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        raise ValueError("Look direction parallel to up")
    r /= np.linalg.norm(r)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([r, np.cross(f, r), f], axis=1)
    pose[:3, 3] = position
    return Camera(pose, K, width, height)
