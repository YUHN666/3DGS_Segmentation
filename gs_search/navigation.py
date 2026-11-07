"""Conservative 3-D occupancy + 6-connected A*. No semantic input.

Each retained Gaussian is bounded by its sigma ellipsoid's axis-aligned box,
then inflated by camera radius + margin. Every intersecting voxel is blocked.
This is intentionally conservative and is not a physical occupancy posterior.
"""
import heapq
import itertools
import time
import numpy as np
from scipy import ndimage


class PlanningError(RuntimeError):
    pass


class OccupancyMap:
    def __init__(self, geometry, config):
        self.config = config
        self.lower, self.upper = np.asarray(config["bounds"], float)
        self.voxel = float(config["voxel_size"])
        self.padding = float(config["camera_radius"]) + float(config["margin"])
        if (not np.isfinite([*self.lower, *self.upper, self.voxel, self.padding]).all()
                or np.any(self.upper <= self.lower) or self.voxel <= 0 or self.padding < 0):
            raise ValueError("Invalid navigation bounds or collision parameters")
        self.shape = np.ceil((self.upper - self.lower) / self.voxel).astype(int)
        cells = int(np.prod(self.shape.astype(object)))
        if cells > config.get("max_cells", 2_000_000):
            raise MemoryError(f"Occupancy needs {cells} cells; exceeds max_cells")
        self.sigma = float(config.get("gaussian_sigma", 2))
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("gaussian_sigma must be positive")
        self.keep = geometry.opacity >= config.get("opacity_threshold", .02)
        ext = geometry.axis_extents(self.sigma)[self.keep] + self.padding
        means = geometry.means[self.keep]
        lo, hi = means - ext, means + ext
        overlap = ((hi >= self.lower) & (lo <= self.upper)).all(1)
        lo, hi = lo[overlap], hi[overlap]
        imin = np.maximum(0, np.floor((lo - self.lower) / self.voxel).astype(np.int64))
        imax = np.minimum(self.shape - 1, np.floor((hi - self.lower) / self.voxel).astype(np.int64))
        # Difference volume: O(N + grid size), never allocate N x grid size.
        diff = np.zeros(tuple(self.shape + 1), np.int32)
        for corner in itertools.product([0, 1], repeat=3):
            ijk = np.where(np.array(corner), imax + 1, imin)
            np.add.at(diff, tuple(ijk.T), (-1) ** sum(corner))
        for axis in range(3):
            np.cumsum(diff, axis=axis, out=diff)
        self.occupied = diff[:-1, :-1, :-1] > 0
        # Respect sphere clearance at navigation volume boundary too.
        for axis in range(3):
            x = self.lower[axis] + (np.arange(self.shape[axis]) + .5) * self.voxel
            mask = (x - self.voxel / 2 < self.lower[axis] + self.padding) | (
                x + self.voxel / 2 > self.upper[axis] - self.padding)
            idx = [slice(None)] * 3
            idx[axis] = mask
            self.occupied[tuple(idx)] = True
        self.labels, self.components = ndimage.label(~self.occupied)
        self.metadata = dict(backend="occupancy_astar_3d", bounds=config["bounds"],
                             shape=self.shape.tolist(), cells=cells, occupied_cells=int(self.occupied.sum()),
                             gaussian_sigma=self.sigma, padding=self.padding,
                             opacity_threshold=config.get("opacity_threshold", .02),
                             total_gaussians=len(geometry.means),
                             opacity_filtered=int((~self.keep).sum()),
                             retained_intersecting_bounds=int(overlap.sum()),
                             obstacle_model="inflated sigma-ellipsoid AABBs, all intersecting voxels",
                             geometric_free_space_is_not_visual_coverage=True)

    def index(self, position):
        p = np.asarray(position, float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise PlanningError("Position must be finite xyz")
        if np.any(p < self.lower) or np.any(p >= self.upper):
            raise PlanningError(f"Outside navigation bounds: {p.tolist()}")
        return tuple(np.floor((p - self.lower) / self.voxel).astype(int))

    def center(self, index):
        return self.lower + (np.asarray(index) + .5) * self.voxel

    def require_free(self, position, name="position"):
        i = self.index(position)
        if self.occupied[i]:
            raise PlanningError(f"{name} is in inflated obstacle/boundary voxel: {np.asarray(position).tolist()}")
        return i

    def snap(self, position):
        """Explicit opt-in only. Caller must log original + replacement."""
        free = np.argwhere(~self.occupied)
        if not len(free):
            raise PlanningError("No free voxel")
        points = self.lower + (free + .5) * self.voxel
        return points[np.argmin(np.sum((points - position) ** 2, axis=1))]

    def segment_free(self, a, b):
        """Exact voxel supercover via all grid-plane crossing events.

        Test open intervals and their endpoints. At a face/edge/corner all
        adjacent closed voxels are tested, preventing diagonal corner cutting.
        Epsilon is used only to include touching voxels, never to skip one.
        """
        a, b = np.asarray(a, float), np.asarray(b, float)
        try:
            self.require_free(a)
            self.require_free(b)
        except PlanningError:
            return False
        d = b - a
        events = [0., 1.]
        for axis in range(3):
            if abs(d[axis]) < 1e-12:
                continue
            planes = self.lower[axis] + np.arange(1, self.shape[axis]) * self.voxel
            ts = (planes - a[axis]) / d[axis]
            events.extend(ts[(ts > 0) & (ts < 1)].tolist())
        events = np.unique(events)
        ts = np.r_[events, (events[:-1] + events[1:]) / 2]
        coords = (a[None] + ts[:, None] * d - self.lower) / self.voxel
        for offset in itertools.product([-1e-8, 1e-8], repeat=3):
            ijk = np.floor(coords + offset).astype(int)
            if np.any(ijk < 0) or np.any(ijk >= self.shape):
                return False
            if self.occupied[tuple(ijk.T)].any():
                return False
        return True

    def validate_path(self, path):
        p = np.asarray(path, float)
        if p.ndim != 2 or p.shape[1] != 3 or len(p) < 1:
            raise PlanningError("Invalid path array")
        for x in p:
            self.require_free(x, "path point")
        for a, b in zip(p[:-1], p[1:]):
            if not self.segment_free(a, b):
                raise PlanningError(f"Continuous segment collision: {a.tolist()} -> {b.tolist()}")
        return True

    def reachable_points(self, position, count, rng):
        label = self.labels[self.require_free(position)]
        indices = np.argwhere(self.labels == label)
        if not len(indices):
            return np.empty((0, 3))
        chosen = rng.choice(len(indices), min(count, len(indices)), replace=False)
        return self.lower + (indices[chosen] + .5) * self.voxel


class AStarPlanner:
    backend = "occupancy_astar_3d"

    def __init__(self, grid):
        self.grid = grid
        self.last_metadata = {}

    def plan(self, start, goal):
        g = self.grid
        si, gi = g.require_free(start, "start"), g.require_free(goal, "goal")
        if g.labels[si] != g.labels[gi]:
            raise PlanningError("Goal unreachable: disconnected free-space components")
        started = time.perf_counter()
        if g.segment_free(start, goal):
            path = np.array([start, goal], float)
        else:
            queue = [(0., 0., si)]
            costs, parent = {si: 0.}, {}
            expanded = 0
            offsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
            while queue:
                _, cost, current = heapq.heappop(queue)
                if cost != costs[current]:
                    continue
                if current == gi:
                    break
                expanded += 1
                if expanded > g.config.get("max_expansions", 500_000):
                    raise PlanningError("A* expansion budget exhausted")
                for off in offsets:
                    nb = tuple(current[k] + off[k] for k in range(3))
                    if any(nb[k] < 0 or nb[k] >= g.shape[k] for k in range(3)) or g.occupied[nb]:
                        continue
                    new = cost + 1
                    if new < costs.get(nb, float("inf")):
                        costs[nb], parent[nb] = new, current
                        heapq.heappush(queue, (new + sum(abs(nb[k]-gi[k]) for k in range(3)), new, nb))
            else:
                raise PlanningError("No collision-free route")
            indices = [gi]
            while indices[-1] != si:
                indices.append(parent[indices[-1]])
            path = np.vstack([start, *[g.center(i) for i in reversed(indices)], goal])
            g.validate_path(path)
            # Greedy line-of-sight shortcut, always followed by full validation.
            short, i = [path[0]], 0
            while i < len(path)-1:
                j = len(path)-1
                while j > i+1 and not g.segment_free(path[i], path[j]):
                    j -= 1
                short.append(path[j])
                i = j
            path = np.array(short)
        path = interpolate(path, g.config.get("move_step", .15))
        g.validate_path(path)
        self.last_metadata = dict(backend=self.backend, planning_seconds=time.perf_counter()-started,
                                  continuous_collision_validated=True)
        return path


def interpolate(path, step):
    if step <= 0:
        raise ValueError("move_step must be positive")
    result = [np.asarray(path[0], float)]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b-a)/step)))
        result.extend(a + (b-a) * (i/n) for i in range(1, n+1))
    return np.array(result)
