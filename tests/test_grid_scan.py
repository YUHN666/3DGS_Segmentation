"""Fixed-camera geometry/control checks. Scripted models are not real inference."""
import json
from pathlib import Path
import numpy as np
import pytest
from gs_search.camera import Camera
from gs_search.grid_scan import FixedGridScan,expand_grid_cameras
from gs_search.controller import SearchController
from gs_search.navigation import AStarPlanner,PlanningError
from gs_search.visualization import VisualizationSession
from test_controller import setup


def layout_for(camera):
    return dict(width=camera.width,height=camera.height,K=camera.K.tolist(),up=[0,-1,0],forward=[0,0,1],
                yaw_degrees=list(range(0,360,45)),
                camera_positions=[dict(name=f'p{i:02d}',position=[float(x),.1,float(z)])
                                  for i,(x,z) in enumerate((x,z) for z in np.linspace(-2,2,4) for x in np.linspace(-2,2,4))])


def policy_for(controller,tmp_path,layout=None):
    from gs_search.scene import Geometry
    g=Geometry(np.empty((0,3)),np.empty((0,3)),np.empty((0,4)),np.empty(0))
    data=layout_for(controller.env.camera) if layout is None else layout
    path=tmp_path/'positions.json';path.write_text(json.dumps(data))
    p=FixedGridScan(g,controller.env.grid,dict(coverage_samples=0,free_coverage_samples=8,
                   grid128=dict(positions_file=str(path),position_order='serpentine')),dict(world_up=[0,-1,0]))
    p.initialize(controller.env.camera)
    return p


def test_camera_expansion_matches_analytic_cardinal_directions():
    # Independent basis check; no private reference repository or scene files.
    from gs_search.camera import look_at
    camera=look_at([0,0,0],[0,0,1],[0,-1,0],[[100,0,80],[0,100,60],[0,0,1]],160,120)
    layout=layout_for(camera)
    views=expand_grid_cameras(layout)
    assert len(views)==16 and all(len(v)==8 for v in views)
    expected=[[0,0,1],[-1,0,0],[0,0,-1],[1,0,0]]
    for actual,forward in zip(views[0][::2],expected):
        np.testing.assert_allclose(actual.c2w[:3,2],forward,atol=1e-10)
        np.testing.assert_allclose(actual.position,layout['camera_positions'][0]['position'])
        np.testing.assert_allclose(actual.K,camera.K)


def test_fixed_positions_angles_order_and_no_planning_as_observation(tmp_path):
    c=setup(tmp_path,True);p=policy_for(c,tmp_path)
    assert p.order==[0,1,2,3,7,6,5,4,8,9,10,11,15,14,13,12]
    planned,path,decision=p.explore(c.env.camera,c.memory,AStarPlanner(c.env.grid),100)
    assert decision['grid_view']==0 and decision['station_id']==0
    assert p.scan_snapshot()['views_observed']==0
    for cameras in p.views:
        assert all(np.allclose(cam.position,cameras[0].position) for cam in cameras)
        for a,b in zip(cameras,cameras[1:]):
            assert np.degrees(np.arccos(np.clip(a.c2w[:3,2]@b.c2w[:3,2],-1,1)))==pytest.approx(45)


def test_obstacle_positions_are_rejected_without_runtime_snapping(tmp_path):
    c=setup(tmp_path,True)
    data=layout_for(c.env.camera);data['camera_positions'][0]['position']=[-3,.1,-3]
    with pytest.raises(PlanningError): policy_for(c,tmp_path,data)


def test_budget_is_partial_and_candidate_does_not_change_fixed_schedule(tmp_path):
    c=setup(tmp_path,False);p=policy_for(c,tmp_path)
    config=c.config;config['search'].update(max_observations=5,max_path_length=100,max_seconds=90)
    controller=SearchController(c.env,c.planner,c.grounder,c.verifier,p,c.artifacts,config)
    r=controller.run()
    assert r['error'] is None and r['observations']==5 and r['vlm_grounding_calls']==5
    assert r['scan_status']=='PARTIAL' and r['grid_scan']['views_observed']==5
    assert r['status']=='INCOMPLETE'
    assert not any(e.get('state')=='VERIFY' for e in c.artifacts.events if e['kind']=='state_transition')
    ids=[e['coverage']['grid_scan']['yaw_degrees'] for e in c.artifacts.events if e['kind']=='observation']
    assert ids==[0,45,90,135,180]
    assert not controller.confirmed_ids


def test_complete_128_only_detects_scheduled_views_with_video(tmp_path):
    c=setup(tmp_path,True);p=policy_for(c,tmp_path)
    c.config['search'].update(max_observations=128,max_path_length=100,max_seconds=120)
    c.artifacts.visualization=VisualizationSession(c.artifacts.path,c.config,
        dict(fps=2,camera_speed=3,angular_speed=180,export_video=False))
    controller=SearchController(c.env,c.planner,c.grounder,c.verifier,p,c.artifacts,c.config)
    r=controller.run()
    assert r['error'] is None,r['error']
    assert r['observations']==r['vlm_grounding_calls']==128
    assert r['render_count']>128
    assert r['stop_reason']=='fixed_grid_128_completed'
    assert r['scan_status']=='COMPLETE' and r['grid_scan']['positions_complete']==16
    assert r['coverage_status']=='NOT_MEASURED' and r['status']=='INCOMPLETE'
    ids=[x for station in r['grid_scan']['stations'] for x in station['view_observation_ids']]
    assert sorted(ids)==list(range(128))
    c.env.grid.validate_path([t['position'] for t in c.env.trajectory])
    # All 8 views at the same first position lack a translation baseline, even
    # though they are distinct observations.
    observations=[e for e in c.artifacts.events if e['kind']=='observation']
    positions=[np.array(e['camera']['camera_to_world'])[:3,3] for e in observations[:8]]
    assert all(np.allclose(p,positions[0]) for p in positions)


def test_intrinsic_mismatch_and_out_of_sequence_rgb_are_explicit_errors(tmp_path):
    c=setup(tmp_path,True);p=policy_for(c,tmp_path)
    with pytest.raises(RuntimeError,match='outside the next scheduled'):
        p.record_arrived(c.env.camera,np.zeros((120,160,3),np.uint8))
    data=layout_for(c.env.camera);data['K'][0][0]=300
    with pytest.raises(ValueError,match='intrinsics'):
        policy_for(c,tmp_path,data)
