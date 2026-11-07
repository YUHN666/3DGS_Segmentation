"""Gaussian PLY membership/export tests with explicitly synthetic masks/scenes."""
import json
import numpy as np
import pytest
from PIL import Image
from plyfile import PlyData,PlyElement
from scipy.spatial.transform import Rotation
from gs_search.camera import Camera
from gs_search.gaussian_projection import export_gaussian_ply,options,select_gaussians,sha256
from gs_search.scene import load_scene


def fixture_run(tmp_path):
    names=['x','y','z',*[f'f_dc_{i}' for i in range(3)],*[f'f_rest_{i}' for i in range(9)],
           'opacity',*[f'scale_{i}' for i in range(3)],*[f'rot_{i}' for i in range(4)]]
    vertex=np.zeros(6,dtype=[(n,'f4') for n in names]+[('untrusted_semantic','i4')])
    vertex['x']=[0,0,.8,0,-.3,np.nan];vertex['z']=[1,1.5,1,1,1,1]
    vertex['rot_0']=1;vertex['opacity']=3;vertex['opacity'][3]=-20
    for i in range(3): vertex[f'scale_{i}']=np.log(.01);vertex[f'f_dc_{i}']=.2
    for i in range(9): vertex[f'f_rest_{i}']=.1
    vertex['untrusted_semantic']=99
    source=tmp_path/'original.ply';PlyData([PlyElement.describe(vertex,'vertex')],comments=['keep me']).write(source)
    config=dict(scene=dict(path=str(source),world_scale=2,max_invalid_fraction=.2),
                gaussian_projection=dict(chunk_size=2),target=dict(description='TEST TARGET'))
    root=tmp_path/'run';root.mkdir();(root/'masks').mkdir();(root/'geometry').mkdir()
    c=Camera(np.eye(4),[[10,0,10],[0,10,10],[0,0,1]],20,20)
    (root/'cameras.json').write_text(json.dumps(dict(cameras=[c.as_dict(),c.as_dict()])))
    mask=np.zeros((20,20),np.uint8);mask[10,10]=255;Image.fromarray(mask).save(root/'masks'/'accepted.png')
    for i in [0,1]: np.savez(root/'geometry'/f'{i:04d}.npz',expected_z=np.full((20,20),2.),alpha=np.ones((20,20)))
    def candidate(i,obs):
        return dict(id=f'candidate_{i:04d}',observation_id=obs,status='accepted_single_view_vlm',
                    semantic_review=dict(matches_target=True),localization=dict(status='LOCALIZED',mask='masks/accepted.png'))
    candidates=[candidate(0,0),candidate(1,0),candidate(2,1),candidate(3,1)]
    candidates[-1]['semantic_review']['matches_target']=False
    return root,config,candidates,vertex


def test_visibility_uses_camera_pose_mask_depth_opacity_and_bounded_chunks():
    pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('y',70,degrees=True).as_matrix();pose[:3,3]=[3,4,5]
    c=Camera(pose,[[10,0,10],[0,10,10],[0,0,1]],20,20)
    pc=np.array([[0,0,2],[0,0,3],[1,0,2],[0,0,-1],[0,0,2],[np.nan,0,2]])
    means=pc@pose[:3,:3].T+pose[:3,3];opacity=np.array([1,1,1,1,0,1])
    mask=np.zeros((20,20),bool);mask[10,10]=True
    cfg=options(dict(gaussian_projection=dict(chunk_size=1)))
    # Use a broad 2x2 mask to avoid relying on exact rounding at the center pixel.
    mask[9:11,9:11]=True
    selected=select_gaussians(means,opacity,c,mask,np.full(mask.shape,2.),np.ones(mask.shape),cfg)
    np.testing.assert_array_equal(selected,[0])
    assert not len(select_gaussians(means,opacity,c,mask,np.full(mask.shape,2.),np.zeros(mask.shape),cfg))


def test_export_preserves_raw_3dgs_fields_source_rows_and_distinct_view_votes(tmp_path):
    root,config,candidates,raw=fixture_run(tmp_path);original_hash=sha256(config['scene']['path'])
    result=export_gaussian_ply(root,config,candidates);dest=root/result['directory']
    assert result['selected_gaussians']==1 and result['source_unchanged']
    assert len(result['skipped'])==1 and result['invalid_geometry_excluded']==1
    assert sha256(config['scene']['path'])==original_hash
    labels=PlyData.read(dest/'scene_target_labels.ply')['vertex'].data
    highlight=PlyData.read(dest/'scene_target_highlighted.ply')['vertex'].data
    target=PlyData.read(dest/'target_gaussians.ply')['vertex'].data
    assert len(labels)==len(raw) and len(target)==1
    for name in raw.dtype.names:
        np.testing.assert_array_equal(labels[name],raw[name])
        np.testing.assert_array_equal(target[name],raw[name][[0]])
        if not name.startswith(('f_dc_','f_rest_')): np.testing.assert_array_equal(highlight[name],raw[name])
        else: np.testing.assert_array_equal(highlight[name][1:],raw[name][1:])
    np.testing.assert_array_equal(labels['search_target'],[1,0,0,0,0,0])
    assert labels['search_view_count'][0]==2  # two candidates in observation 0 count only once
    np.testing.assert_array_equal(np.load(dest/'target_gaussian_indices.npy'),[0])
    with np.load(dest/'candidate_gaussian_indices.npz') as mapping:
        assert len(mapping.files)==3 and 'candidate_0003' not in mapping.files
    assert load_scene(dict(path=str(dest/'target_gaussians.ply'))).metadata['sh_degree']==1
    assert 'keep me' in PlyData.read(dest/'scene_target_labels.ply').comments


def test_nonaccepted_same_class_masks_never_label_scene(tmp_path):
    root,config,candidates,raw=fixture_run(tmp_path)
    for c in candidates: c['status']='uncertain_vlm'
    r=export_gaussian_ply(root,config,candidates);dest=root/r['directory']
    assert r['status']=='EMPTY' and r['selected_gaussians']==0
    v=PlyData.read(dest/'scene_target_highlighted.ply')['vertex'].data
    assert not v['search_target'].any()
    for name in raw.dtype.names: np.testing.assert_array_equal(v[name],raw[name])


def test_projection_failure_keeps_diagnostic_and_does_not_overwrite_source(tmp_path):
    root,config,candidates,_=fixture_run(tmp_path);before=sha256(config['scene']['path'])
    config['gaussian_projection']['max_export_bytes']=1
    with pytest.raises(MemoryError): export_gaussian_ply(root,config,candidates)
    report=json.loads(next((root/'gaussian_projection').glob('*/projection_report.json')).read_text())
    assert report['status']=='ERROR' and report['error']['type']=='MemoryError'
    assert sha256(config['scene']['path'])==before
