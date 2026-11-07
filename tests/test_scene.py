import numpy as np
import pytest
from plyfile import PlyData,PlyElement
from gs_search.scene import load_scene
from gs_search.cli import with_fov
from gs_search.camera import Camera


def write_ply(path, bad=False):
    names = ['x','y','z',*[f'f_dc_{i}' for i in range(3)],*[f'f_rest_{i}' for i in range(9)],
             'opacity',*[f'scale_{i}' for i in range(3)],*[f'rot_{i}' for i in range(4)],'semantic_label']
    v=np.zeros(2,dtype=[(n,'f4') for n in names]);v['rot_0']=2
    v['scale_0']=np.log(.1);v['scale_1']=np.log(.2);v['scale_2']=np.log(.3)
    v['semantic_label']=9876
    for i in range(9): v[f'f_rest_{i}']=i
    if bad: v['x'][0]=np.nan
    PlyData([PlyElement.describe(v,'vertex')],text=False).write(path)


def test_activations_quaternion_channel_order_and_semantic_exclusion(tmp_path):
    p=tmp_path/'scene.ply';write_ply(p)
    s=load_scene({'path':str(p),'world_scale':2})
    assert np.allclose(s.geometry.scales[0],[.2,.4,.6])
    assert np.allclose(s.geometry.opacity,.5)
    assert np.allclose(s.geometry.quats[0],[1,0,0,0])
    assert np.allclose(s.sh[0,1:,:],[[0,3,6],[1,4,7],[2,5,8]])
    assert not hasattr(s,'semantic_label') and not hasattr(s.geometry,'semantic_label')
    assert s.metadata['ignored_extra_fields']==['semantic_label']


def test_invalid_gaussians_fail_without_silent_filter(tmp_path):
    p=tmp_path/'bad.ply';write_ply(p,True)
    with pytest.raises(ValueError,match='Invalid Gaussians'):
        load_scene({'path':str(p)})
    s=load_scene({'path':str(p),'max_invalid_fraction':.5})
    assert s.metadata['invalid_removed']==1


def test_fov_controls_recorded_intrinsics():
    c=Camera(np.eye(4),[[200,0,100],[0,200,80],[0,0,1]],200,160)
    wide=with_fov(c,90)
    assert np.isclose(wide.K[0,0],100)
    with pytest.raises(ValueError): with_fov(c,179)
