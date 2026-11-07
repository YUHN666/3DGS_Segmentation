"""Generate a geometry-only bypass/occlusion scene, no semantic target data.

This is for rendering/navigation regression, not a real object detection claim.
"""
from pathlib import Path
import json
import numpy as np
from plyfile import PlyData, PlyElement


def generate(root):
    rng = np.random.default_rng(4)
    surfaces = []
    # A floor, two side walls and a full-height partition with a bypass at z>3.5.
    for x in np.arange(.1, 5.95, .10):
        for z in np.arange(.1, 4.95, .10):
            surfaces.append([x,2.9,z])
    for y in np.arange(.05,2.95,.10):
        for z in np.arange(.05,3.45,.10):
            surfaces.append([3.,y,z])
        for x in np.arange(.05,5.95,.10):
            surfaces.extend([[x,y,.05],[x,y,4.95]])
    xyz = np.array(surfaces,np.float32)
    names = ['x','y','z',*[f'f_dc_{i}' for i in range(3)],'opacity',*[f'scale_{i}' for i in range(3)],*[f'rot_{i}' for i in range(4)]]
    v = np.zeros(len(xyz),dtype=[(n,'f4') for n in names])
    rgb = rng.uniform(.2,.85,(len(v),3))
    for i,n in enumerate(['x','y','z']):
        v[n] = xyz[:,i]
        v[f'f_dc_{i}'] = (rgb[:,i]-.5)/.28209479177387814
        v[f'scale_{i}'] = np.log(.055)
    v['rot_0'],v['opacity'] = 1.,5.
    path = root/'synthetic'/'point_cloud.ply'
    path.parent.mkdir(parents=True,exist_ok=True)
    PlyData([PlyElement.describe(v,'vertex')],text=False).write(path)
    c = json.loads((root/'configs/search.json').read_text())
    c['scene']['path'] = str(path)
    c['scene']['navigation_enabled'] = True
    c['renderer']['backend'] = 'cpu_reference'
    c['renderer']['device'] = 'cpu'
    c['camera'] = dict(width=400,height=300,start=[1.1,1.5,1.1],direction=[1,0,0],world_up=[0,-1,0],fov_degrees=75)
    c['navigation'].update(backend='occupancy_astar_3d',bounds=[[0,0,0],[6,3,5]],voxel_size=.2,camera_radius=.04,margin=.02)
    (root/'configs/synthetic.json').write_text(json.dumps(c,indent=2))
    print(path)


if __name__ == '__main__':
    generate(Path(__file__).resolve().parents[1])
