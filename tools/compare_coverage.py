"""Offline evaluation of actual saved RGB, never a search prior or new inference."""
import argparse
import json
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
from gs_search.cli import load_config,initial_camera
from gs_search.scene import load_scene
from gs_search.navigation import OccupancyMap
from gs_search.coverage import CoveragePolicy
from gs_search.camera import Camera
from gs_search.artifacts import write_json

if __name__=='__main__':
    Path('artifacts').mkdir(exist_ok=True)
    parser=argparse.ArgumentParser();parser.add_argument('--baseline',required=True);parser.add_argument('--run',required=True)
    parser.add_argument('--limit',type=int,default=30);parser.add_argument('--output',default='artifacts/coverage_comparison.json');args=parser.parse_args()
    run=Path(args.run).resolve();baseline=Path(args.baseline).resolve()
    c=load_config(run/'config.json');scene=load_scene(c['scene']);grid=OccupancyMap(scene.geometry,c['navigation'])
    policy=CoveragePolicy(scene.geometry,grid,c['policy'],c['camera']);policy.initialize(initial_camera(c))
    events=[json.loads(x) for x in (run/'decisions.jsonl').read_text().splitlines()]
    initial=next(e['coverage'] for e in events if e['kind']=='coverage_initialized')
    assert np.allclose(initial['points'],policy.patches)
    base=[json.loads(x) for x in (baseline/'decisions.jsonl').read_text().splitlines()]
    base=[x for x in base if x['kind']=='observation'][:args.limit]
    for e in base:
        rgb=np.asarray(Image.open(baseline/e['rgb']).convert('RGB'))
        policy.record_arrived(Camera.from_dict(e['camera']),rgb)
    current=[x for x in events if x['kind']=='observation'][:args.limit]
    keys=['surface_patches','surface_patches_seen','surface_patches_effective',
          'sampled_potentially_effective_patches','effective_fraction_of_sampled_observable']
    old=policy.summary();new=current[-1]['coverage']
    result=dict(baseline=str(baseline),run=str(run),same_observation_budget=len(base)==len(current),
                baseline_observations=len(base),new_observations=len(current),
                baseline_path_length=base[-1]['path_length'],new_path_length=current[-1]['path_length'],
                fixed_patch_sha256=hashlib.sha256(policy.patches.tobytes()).hexdigest(),
                baseline_coverage={k:old[k] for k in keys},new_coverage={k:new[k] for k in keys},
                note='Offline evaluation of actual saved RGB with identical sampled geometry; no semantic model calls; not object recall')
    write_json(args.output,result);print(json.dumps(result,ensure_ascii=False,indent=2))
