"""Offline audit of saved geometry motion and multiview evidence conditions."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from gs_search.scene import load_scene
from gs_search.navigation import OccupancyMap
from gs_search.artifacts import write_json


def audit(path):
    path=Path(path).resolve()
    c=json.loads((path/'config.json').read_text())
    r=json.loads((path/'report.json').read_text())
    events=[json.loads(x) for x in (path/'decisions.jsonl').read_text().splitlines()]
    observations=[x for x in events if x['kind']=='observation']
    assert len(observations)==r['observations']==r.get('vlm_grounding_calls',r.get('detector_calls'))
    assert r['render_count']>=r['observations']
    if r.get('visualization'):
        manifest=json.loads((path/'visualization/manifest.json').read_text())
        assert len(manifest['frames'])==r['visualization']['frame_count']
    scene=load_scene(c['scene']);grid=OccupancyMap(scene.geometry,c['navigation'])
    trajectory=json.loads((path/'trajectory.json').read_text())
    positions=np.array([x['position'] for x in trajectory])
    grid.validate_path(positions)
    if r.get('visualization'):
        grid.validate_path([np.asarray(f['camera']['camera_to_world'])[:3,3] for f in manifest['frames']])
    length=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).sum())
    assert abs(length-r['path_length'])<1e-6
    for o in observations:
        im=np.asarray(Image.open(path/o['rgb']).convert('RGB'))
        assert im.std()>2
        assert o['camera']['width']==im.shape[1] and o['camera']['height']==im.shape[0]
    coverage_checked=False
    grid_scan_checked=False
    if (path/'grid_scan.json').exists():
        scan=json.loads((path/'grid_scan.json').read_text())
        assert scan==r['grid_scan']
        assert scan['views_total']==128 and scan['positions_total']==16
        assert scan['views_observed']==len(observations)
        ids=[i for s in scan['stations'] for i in s['view_observation_ids'] if i>=0]
        assert sorted(ids)==list(range(len(observations)))
        for o,scheduled in zip(observations,scan['cameras']):
            assert np.allclose(o['camera']['camera_to_world'],scheduled['camera_to_world'],atol=1e-6)
            assert np.allclose(o['camera']['K'],scheduled['K'])
        if scan['scan_complete']: assert len(observations)==128
        grid_scan_checked=True
    if (path/'coverage.json').exists():
        coverage=json.loads((path/'coverage.json').read_text())
        assert coverage['summary']==r['coverage']
        initial=next(e['coverage'] for e in events if e['kind']=='coverage_initialized')
        for o in observations:
            delta=o['coverage']['coverage_update'];obs=o['observation_id']
            for i in delta['seen']:
                assert initial['first_seen'][i]==-1
                initial['first_seen'][i]=obs
            for i in delta['effective']:
                assert initial['first_effective'][i]==-1
                initial['first_effective'][i]=obs
            for i,j in delta['station_views']:
                assert initial['stations'][i]['view_first_observed'][j]==-1
                assert np.linalg.norm(np.array(initial['stations'][i]['position'])-np.array(o['camera']['camera_to_world'])[:3,3])<1e-4
                initial['stations'][i]['view_first_observed'][j]=obs
        assert initial['first_seen']==coverage['first_seen']
        assert initial['first_effective']==coverage['first_effective']
        assert [s['view_first_observed'] for s in initial['stations']]==[s['view_first_observed'] for s in coverage['stations']]
        coverage_checked=True
    if r['status']=='FOUND':
        found=[x for x in r['candidates'] if x['status'].startswith('confirmed')]
        assert found
        for candidate in found:
            assert len(candidate['evidence'])>=2
            for link in candidate['association_links']:
                assert link['baseline']>=c['association']['min_baseline']
                assert link['inliers']>=c['association']['min_feature_inliers']
            if c['target'].get('require_full_description'):
                assert candidate['semantic_review']['matches_target'] is True
    correction_path=path/'review_correction.json'
    correction=json.loads(correction_path.read_text()) if correction_path.exists() else None
    return dict(run_dir=str(path),status=(correction['review_status'] if correction else r['status']),
                original_status=r['status'],manual_review=correction,observations=len(observations),
                path_length=length,continuous_saved_trajectory_valid=True,
                multiview_confirmation_conditions_checked=r['status']=='FOUND',
                coverage_observation_timestamps_checked=coverage_checked,
                fixed_grid_schedule_checked=grid_scan_checked,
                semantic_accuracy_established=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_dirs',nargs='+');args=p.parse_args()
    result=[audit(x) for x in args.run_dirs]
    dest=Path(__file__).resolve().parents[1]/'artifacts/run_audit.json'
    dest.parent.mkdir(exist_ok=True,parents=True)
    write_json(dest,result);print(json.dumps(result,ensure_ascii=False,indent=2))
