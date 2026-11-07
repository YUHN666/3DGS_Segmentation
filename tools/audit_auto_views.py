"""Audit real geometry-only acquisition, cameras and cached RGB/expected-Z."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from gs_search.artifacts import write_json
from gs_search.auto_views import validate_binding
from gs_search.camera import Camera
from gs_search.navigation import OccupancyMap
from gs_search.offline import offline_cameras
from gs_search.scene import load_scene


def main():
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output',required=True);args=p.parse_args()
    root=Path(args.run).resolve();config=json.loads((root/'config.json').read_text())
    report=json.loads((root/'report.json').read_text());layout=json.loads((root/'auto_cameras/layout.json').read_text())
    validate_binding(layout,config['scene'])
    assert report['status']=='PREPARED' and report['preparation_only'] and report['processing_complete']
    assert report['vlm_calls']==report['sam2_calls']==0 and report['render_count']==128
    assert report['grounding_status']=='NOT_RUN' and report['target_found'] is False
    scene=load_scene(config['scene']);grid=OccupancyMap(scene.geometry,config['navigation'])
    cameras=json.loads((root/'cameras.json').read_text())['cameras'];planned=offline_cameras(config)
    assert len(cameras)==len(planned)==128
    stds=[];foreground=[];depth_fraction=[]
    for i,(actual,expected) in enumerate(zip(cameras,planned)):
        camera=Camera.from_dict(actual);grid.require_free(camera.position)
        np.testing.assert_allclose(actual['camera_to_world'],expected['camera_to_world'],atol=1e-8)
        np.testing.assert_allclose(actual['K'],expected['K'],atol=1e-8)
        rgb=np.asarray(Image.open(root/'rgb'/f'{i:04d}.png').convert('RGB'))
        assert rgb.shape==(camera.height,camera.width,3)
        nonblack=float((rgb.max(2)>5).mean());assert nonblack>.01
        stds.append(float(rgb.std()));foreground.append(nonblack)
        with np.load(root/'geometry'/f'{i:04d}.npz') as data:
            z,alpha=data['expected_z'],data['alpha'];assert z.shape==alpha.shape==rgb.shape[:2]
            valid=np.isfinite(z)&(z>config['renderer']['near'])&np.isfinite(alpha)&(alpha>.1)
            fraction=float(valid.mean());assert fraction>.01;depth_fraction.append(fraction)
    result=dict(status='PASS',run=str(root),scene=config['scene']['path'],gaussians=scene.metadata['count'],
        safe_positions=16,rendered_views=128,vlm_calls=0,sam2_calls=0,all_camera_positions_collision_free=True,
        all_poses_and_intrinsics_match_generated_layout=True,scene_binding_valid=True,
        min_rgb_std=min(stds),low_texture_views=sum(s<2 for s in stds),
        min_nonblack_fraction=min(foreground),min_valid_depth_alpha_fraction=min(depth_fraction),
        preparation=report['camera_preparation'],timing_seconds=report['timing_seconds'],
        limitation='Geometry feasibility and rendering consistency, not exhaustive visual/semantic coverage')
    write_json(args.output,result);write_json(root/'auto_cameras/validation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='preparation'},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
