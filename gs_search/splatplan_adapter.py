"""Upstream SplatPlan corridor + Bezier optimizer, with a checked A* seed.

The upstream seed silently snaps endpoints and its UnfoldNd grid has unbounded
memory growth. We replace only that seed interface, not corridor construction
or spline optimization. Returned polylines undergo our independent validator.
No Nerfstudio loader, Splat-Loc, semantic query or robot runtime is imported.
"""
from pathlib import Path
from types import SimpleNamespace
import copy
import sys
import traceback
import numpy as np
from scipy import ndimage
from .navigation import AStarPlanner, PlanningError, interpolate


class SplatPlanAdapter:
    backend = "splatplan_checked_seed"

    def __init__(self, geometry, grid, config):
        import torch
        repo = Path(config.get("repo", "third_party/splatnav")).resolve()
        if not (repo / "splatplan/splatplan.py").is_file():
            raise RuntimeError(f"Splat-Nav checkout missing: {repo}")
        sys.path.insert(0, str(repo))
        from splatplan.splatplan import SplatPlan
        from splatplan.spline_utils import SplinePlanner
        from polytopes.collision_set import GSplatCollisionSet
        device = torch.device(config.get("device", "cuda:0"))
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Upstream SplatPlan calls CUDA synchronize; CUDA required")
        self.grid, self.torch, self.device = grid, torch, device
        self.config = config
        self.seed = AStarPlanner(grid)
        def tensor(x):
            return torch.as_tensor(x, device=device, dtype=torch.float32)
        # A geometric adapter; no color/label data are required by the planner.
        g = SimpleNamespace(means=tensor(geometry.means[grid.keep]),
                            scales=tensor(geometry.scales[grid.keep] * grid.sigma),
                            rots=tensor(geometry.quats[grid.keep]))
        # Construct the real upstream class, substituting its voxel seed as
        # documented above. The same upstream generate_path executes unchanged.
        p = SplatPlan.__new__(SplatPlan)
        p.gsplat, p.device = g, device
        p.radius = grid.padding
        p.vmax, p.amax = config.get("vmax", .5), config.get("amax", .5)
        p.collision_set = GSplatCollisionSet(g, p.vmax, p.amax, p.radius, device)
        seed = self.seed
        p.gsplat_voxel = SimpleNamespace(create_path=lambda a,b: seed.plan(a.cpu().numpy(), b.cpu().numpy()).astype(np.float32))
        p.spline_planner = SplinePlanner(spline_deg=6, N_sec=20, device=device)
        p.times_cbf, p.times_qp, p.times_prune = [], [], []
        self.planner = p
        self.last_metadata = {}

    def plan(self, start, goal):
        self.grid.require_free(start, "start")
        self.grid.require_free(goal, "goal")
        if np.linalg.norm(np.asarray(start)-goal) < 1e-6:
            return np.array([start], float)
        t = self.torch
        attempts = []
        last_error = None
        # The upstream ellipsoid corridor is less conservative than our voxel
        # AABBs. If its smoothed polyline fails the final voxel check, retry the
        # SAME upstream corridor/Bezier planner with a higher-clearance seed.
        # Never return the A* seed as if it were an optimized SplatPlan path.
        for clearance in self.config.get("seed_clearance_voxels", [0, 1, 2]):
            try:
                seed_grid = self.grid
                if clearance:
                    seed_grid = copy.copy(self.grid)
                    seed_grid.occupied = ndimage.binary_dilation(self.grid.occupied, iterations=int(clearance),
                                                                 structure=np.ones((3,3,3), bool), border_value=1)
                    seed_grid.labels, seed_grid.components = ndimage.label(~seed_grid.occupied)
                seed = AStarPlanner(seed_grid)
                self.planner.gsplat_voxel = SimpleNamespace(create_path=lambda a,b: seed.plan(
                    a.cpu().numpy(), b.cpu().numpy()).astype(np.float32))
                with t.inference_mode():
                    data = self.planner.generate_path(t.tensor(start, dtype=t.float32, device=self.device),
                                                      t.tensor(goal, dtype=t.float32, device=self.device))
                if not data["feasible"]:
                    raise PlanningError("SplatPlan optimizer infeasible")
                path = np.asarray(data["traj"], float)[:, :3]
                if not np.allclose(path[0], start, atol=1e-4) or not np.allclose(path[-1], goal, atol=1e-4):
                    raise PlanningError("SplatPlan changed endpoints")
                path[0], path[-1] = start, goal
                path = interpolate(path, self.grid.config.get("move_step", .15))
                self.grid.validate_path(path)
                attempts.append(dict(seed_clearance_voxels=clearance, status="validated"))
                self.last_metadata = dict(backend=self.backend, upstream_result=data, attempts=attempts,
                                          seed="conservative occupancy A*; replaces upstream GSplatVoxel",
                                          continuous_collision_validated=True)
                return path
            except Exception as exc:
                attempts.append(dict(seed_clearance_voxels=clearance, status="rejected", error=repr(exc),
                                     traceback=traceback.format_exc()))
                last_error = exc
        self.last_metadata = dict(backend=self.backend, attempts=attempts, continuous_collision_validated=False)
        raise PlanningError(f"SplatPlan rejected all seed clearances: {attempts[-1]['error']}") from last_error
