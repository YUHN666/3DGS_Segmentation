"""Time-sampled camera motion on the actual planned polyline."""
from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .camera import Camera


@dataclass
class MotionStep:
    camera: Camera
    duration: float
    arrived: bool


def validate_options(options):
    for key, default, upper in [('fps', 12, 60), ('camera_speed', .35, 100), ('angular_speed', 45, 720)]:
        value = float(options.get(key, default))
        if not math.isfinite(value) or not 0 < value <= upper:
            raise ValueError(f'{key} must be positive and <= {upper}')
    if not 1 <= int(options.get('max_frames', 10000)) <= 100000:
        raise ValueError('max_frames must be in [1,100000]')
    if not math.isfinite(float(options.get('max_disk_mb', 1024))) or float(options.get('max_disk_mb', 1024)) <= 0:
        raise ValueError('max_disk_mb must be positive')


def between(a, b, options, arrived=True):
    """Interpolate a single executed segment, including a pure camera turn.

    Camera speed is in scene units/second; angular speed is degrees/second.
    Emitted samples never extrapolate beyond the input segment.
    """
    fps = float(options.get('fps', 12))
    angle = np.degrees((Rotation.from_matrix(a.c2w[:3,:3]).inv()*
                        Rotation.from_matrix(b.c2w[:3,:3])).magnitude())
    duration = max(np.linalg.norm(b.position-a.position)/options.get('camera_speed', .35),
                   angle/options.get('angular_speed', 45))
    n = max(1, int(math.ceil(duration*fps-1e-8)))
    # Bound work for malformed replay trajectories before allocating arrays.
    if n > options.get('max_frames', 10000):
        raise ValueError('One motion segment exceeds the recording frame limit')
    slerp = Slerp([0,1], Rotation.from_matrix(np.stack([a.c2w[:3,:3], b.c2w[:3,:3]])))
    for i in range(1, n+1):
        alpha = i/n
        pose = np.eye(4)
        pose[:3,:3] = slerp(alpha).as_matrix()
        pose[:3,3] = a.position+(b.position-a.position)*alpha
        K = a.K+(b.K-a.K)*alpha
        yield MotionStep(Camera(pose,K,b.width,b.height),1/fps,arrived and i==n)


def along_path(initial, target, path, options):
    """Keep each original path vertex; densify only its actual segments."""
    path = np.asarray(path, float)
    if len(path) == 1:
        yield from between(initial,target,options)
        return
    slerp = Slerp([0,1],Rotation.from_matrix(np.stack([initial.c2w[:3,:3],target.c2w[:3,:3]])))
    previous = initial
    for i,position in enumerate(path[1:],1):
        alpha = i/(len(path)-1)
        pose = np.eye(4)
        pose[:3,:3],pose[:3,3] = slerp(alpha).as_matrix(),position
        end = Camera(pose,initial.K+(target.K-initial.K)*alpha,target.width,target.height)
        yield from between(previous,end,options,arrived=i==len(path)-1)
        previous = end
