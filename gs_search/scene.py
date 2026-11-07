"""Graphdeco and InteriorGS compact PLY. Only explicitly allowed fields are read.

No semantic/instance fields are exposed, even when present in an input PLY.
Geometry and appearance stay in the environment process, outside perception.
"""
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation
from scipy.special import expit


@dataclass
class Geometry:
    means: np.ndarray
    scales: np.ndarray
    quats: np.ndarray  # wxyz
    opacity: np.ndarray

    def axis_extents(self, sigma):
        R = Rotation.from_quat(self.quats[:, [1, 2, 3, 0]]).as_matrix()
        return sigma * np.sqrt(np.sum(R ** 2 * self.scales[:, None, :] ** 2, axis=2))


@dataclass
class Scene:
    geometry: Geometry
    sh: np.ndarray  # N,K,3; Graphdeco f_rest is stored in channel-major order
    metadata: dict
    source_indices: np.ndarray | None = None  # original PLY rows, including when invalid rows were filtered


def load_scene(config):
    path = Path(config["path"])
    max_bytes = config.get("max_file_bytes", 2_000_000_000)
    if path.stat().st_size > max_bytes:
        raise MemoryError(f"PLY exceeds max_file_bytes={max_bytes}: {path}")
    v = PlyData.read(path, mmap=True)["vertex"].data
    n = len(v)
    if n > config.get("max_gaussians", 7_000_000):
        raise MemoryError(f"{n} Gaussians exceeds max_gaussians; no silent subsampling")
    names = set(v.dtype.names)
    required = ["x", "y", "z", "opacity", *[f"scale_{i}" for i in range(3)],
                *[f"rot_{i}" for i in range(4)], *[f"f_dc_{i}" for i in range(3)]]
    if set(required) - names:
        raise ValueError(f"Not a supported 3DGS PLY; missing {sorted(set(required)-names)}")
    def columns(keys):
        return np.column_stack([v[k] for k in keys]).astype(np.float32)
    xyz = columns(["x", "y", "z"])
    logs = columns([f"scale_{i}" for i in range(3)])
    q = columns([f"rot_{i}" for i in range(4)])
    logits = np.asarray(v["opacity"], dtype=np.float32)
    dc = columns([f"f_dc_{i}" for i in range(3)])[:, None, :]
    rest = sorted((k for k in names if k.startswith("f_rest_")), key=lambda k: int(k[7:]))
    if rest:
        if len(rest) % 3:
            raise ValueError("Malformed spherical harmonics")
        sh = np.concatenate([dc, columns(rest).reshape(n, 3, -1).transpose(0, 2, 1)], 1)
    else:
        sh = dc
    degree = math.isqrt(sh.shape[1]) - 1
    if (degree + 1) ** 2 != sh.shape[1] or degree > 4:
        raise ValueError("Unsupported SH coefficient count")
    valid = (np.isfinite(xyz).all(1) & np.isfinite(logs).all(1) &
             np.isfinite(q).all(1) & (np.linalg.norm(q, axis=1) > 1e-10) &
             np.isfinite(logits) & np.isfinite(sh).all((1, 2)) &
             (logs < 40).all(1) & (logs > -80).all(1))
    bad = int((~valid).sum())
    if bad / max(n, 1) > config.get("max_invalid_fraction", 0):
        raise ValueError(f"Invalid Gaussians {bad}/{n}; adjust max_invalid_fraction explicitly if intended")
    scale = float(config.get("world_scale", 1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("world_scale must be positive")
    xyz, logs, q, logits, sh = [a[valid].copy() for a in (xyz, logs, q, logits, sh)]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    scales = np.exp(logs) * scale
    metadata = dict(path=str(path.resolve()), format="graphdeco_ply", count=n,
                    invalid_removed=bad, world_scale=scale, sh_degree=degree,
                    scale_activation="exp", opacity_activation="sigmoid", quaternion_order="wxyz",
                    ignored_extra_fields=sorted(names - set(required) - set(rest)))
    return Scene(Geometry(xyz * scale, scales, q, expit(logits)), sh, metadata, np.flatnonzero(valid))
