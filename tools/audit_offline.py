"""Audit offline acquisition, semantic gating and saved 3D projection consistency."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from plyfile import PlyData
from gs_search.camera import Camera
from gs_search.artifacts import write_json


def audit(root):
    root=Path(root).resolve();r=json.loads((root/'report.json').read_text())
    cameras=json.loads((root/'cameras.json').read_text())['cameras']
    assert len(cameras)==r['observations']==128
    assert r['navigation_executed'] is False and r['path_length']==0
    assert json.loads((root/'trajectory.json').read_text())==[]
    images=list((root/'rgb').glob('*.png'));assert len(images)==128
    stds=[float(np.asarray(Image.open(p)).std()) for p in images];assert min(stds)>2
    candidates=r['candidates'];accepted=[c for c in candidates if c['status']=='accepted_single_view_vlm']
    checks=[]
    for c in candidates:
        loc=c['localization']
        if c['status']=='accepted_single_view_vlm': assert c['semantic_review']['matches_target'] is True
        else:
            assert c['target_xyz'] is None and loc['status']=='NOT_ATTEMPTED'
        if loc['status']!='LOCALIZED': continue
        camera=Camera.from_dict(cameras[c['observation_id']])
        vertex=PlyData.read(str(root/loc['pointcloud']))['vertex'].data
        points=np.column_stack([vertex[x] for x in ['x','y','z']]).astype(float)
        q=(points-camera.position)@camera.c2w[:3,:3]
        uv=q@camera.K.T;uv=uv[:,:2]/uv[:,2:]
        pixels=np.rint(uv-.5).astype(int)
        assert (pixels>=0).all() and (pixels[:,0]<camera.width).all() and (pixels[:,1]<camera.height).all()
        mask=np.asarray(Image.open(root/loc['mask']))>0
        assert mask[pixels[:,1],pixels[:,0]].all()
        with np.load(root/'geometry'/f"{c['observation_id']:04d}.npz") as data:
            reference=data['expected_z'][pixels[:,1],pixels[:,0]]
            alpha=data['alpha'][pixels[:,1],pixels[:,0]]
        error=float(np.abs(reference-q[:,2]).max());assert error<1e-4,error
        assert (alpha>=.6-1e-6).all()
        np.testing.assert_allclose(np.median(points,axis=0),c['target_xyz'],atol=1e-5)
        checks.append(dict(id=c['id'],points=len(points),max_depth_reprojection_error=error,target_xyz=c['target_xyz']))
    assert bool(accepted)==r['target_found']
    return dict(run=str(root),rgb_views=128,min_rgb_std=min(stds),proposals=len(candidates),
                all_proposals_reviewed=all(bool(c['semantic_review']) for c in candidates),
                vlm_accepted=len(accepted),localized=len(checks),projection_checks=checks,
                navigation_not_executed=True,pointclouds_only_after_vlm=True,
                note='Projection consistency verifies geometry plumbing, not semantic ground truth or calibrated localization accuracy')

if __name__=='__main__':
    Path('artifacts').mkdir(exist_ok=True)
    parser=argparse.ArgumentParser();parser.add_argument('run');parser.add_argument('--output',default='artifacts/offline_audit.json');args=parser.parse_args()
    result=audit(args.run);write_json(args.output,result);print(json.dumps(result,ensure_ascii=False,indent=2))
