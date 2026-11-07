import numpy as np
import pytest
from scipy import ndimage
from gs_search.scene import Geometry
from gs_search.navigation import OccupancyMap, AStarPlanner, PlanningError, interpolate


def wall_scene(sealed=False):
    ys = np.arange(-.2, 3.3, .2)
    zs = np.arange(-.2, 5.3 if sealed else 3.5, .2)
    means = np.array([[3., y, z] for y in ys for z in zs], np.float32)
    return Geometry(means, np.full_like(means, .09), np.tile([1., 0, 0, 0], (len(means), 1)), np.ones(len(means)))


def grid(sealed=False):
    return OccupancyMap(wall_scene(sealed), dict(bounds=[[0, 0, 0], [6, 3, 5]], voxel_size=.2,
                         camera_radius=.04, margin=.02, gaussian_sigma=2, opacity_threshold=.02, move_step=.07))


def test_detour_full_segments_and_interpolation():
    g = grid()
    a, b = np.array([1.1, 1.5, 1.1]), np.array([4.9, 1.5, 1.1])
    assert not g.segment_free(a, b)
    path = AStarPlanner(g).plan(a, b)
    assert g.validate_path(path)
    assert np.max(path[:, 2]) > 3.5
    assert np.allclose(path[0], a) and np.allclose(path[-1], b)
    assert np.linalg.norm(np.diff(path, axis=0), axis=1).max() <= .07+1e-9
    assert g.validate_path(interpolate(path, .025))


def test_unreachable_and_illegal_endpoints():
    g = grid(True)
    planner = AStarPlanner(g)
    with pytest.raises(PlanningError, match="unreachable"):
        planner.plan([1.1, 1.5, 1.1], [4.9, 1.5, 1.1])
    with pytest.raises(PlanningError, match="start"):
        planner.plan([3, 1.5, 1.1], [4.9, 1.5, 1.1])
    with pytest.raises(PlanningError, match="goal"):
        planner.plan([1.1, 1.5, 1.1], [3, 1.5, 1.1])
    with pytest.raises(PlanningError, match="Outside"):
        planner.plan([-1, 1, 1], [1, 1, 1])
    assert not np.allclose(g.snap([3, 1.5, 1.1]), [3, 1.5, 1.1])


def test_corner_touch_and_thin_mid_segment_obstacle():
    g = grid()
    g.occupied[:] = False
    g.occupied[10, 11, 10] = True
    # Crosses an exact grid corner touching an obstacle voxel at that event.
    a, b = g.center([10, 10, 10]), g.center([11, 11, 10])
    assert not g.segment_free(a, b)
    g.occupied[:] = False
    g.occupied[15, 7, 5] = True
    assert not g.segment_free(g.center([3, 7, 5]), g.center([25, 7, 5]))
    assert g.segment_free(g.center([3, 7, 6]), g.center([25, 7, 6]))


def test_memory_guard_and_nan():
    with pytest.raises(MemoryError):
        OccupancyMap(wall_scene(), dict(bounds=[[0,0,0], [100,100,100]], voxel_size=.001,
                                       camera_radius=.1, margin=.1))
    with pytest.raises(PlanningError):
        grid().require_free([float("nan"), 0, 0])
