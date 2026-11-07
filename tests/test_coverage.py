"""Coverage geometry/control tests; detector/FOUND doubles are labelled tests."""
import json
import numpy as np
import pytest
from gs_search.coverage import CoveragePolicy,clear_rays
from gs_search.camera import look_at
from gs_search.navigation import AStarPlanner
from gs_search.visualization import VisualizationSession
from test_navigation import grid
from test_controller import setup


def make_policy(g=None,**overrides):
    g=grid() if g is None else g
    options=dict(station_spacing=2.,max_stations=128,max_patches=512,patch_size=.4,
                 yaw_samples=4,pitch_degrees=[0,-35,30],views_per_visit=2)
    options.update(overrides)
    p=CoveragePolicy(g_geometry(g),g,dict(coverage=options,coverage_samples=0,free_coverage_samples=8,
                       coverage_object_size=.3,min_object_pixels=5),dict(world_up=[0,1,0]))
    camera=look_at([1.1,1.5,1.1],[1,0,0],[0,1,0],[[100,0,80],[0,100,60],[0,0,1]],160,120)
    p.initialize(camera)
    return p,camera


def g_geometry(g):
    from test_navigation import wall_scene
    return wall_scene()


def test_visibility_is_uninflated_and_wall_occludes_rays():
    p,c=make_policy()
    assert p.grid.padding>0 and p.visibility_grid.padding==0
    raw=p.visibility_grid
    clear=clear_rays(c.position,np.array([[4.9,1.5,1.1],[1.1,1.5,4.1]]),raw.occupied,raw.lower,raw.voxel)
    assert clear.tolist()==[False,True]
    assert not p.grid.segment_free(c.position,[4.9,1.5,1.1])
    assert p.patch_metadata['cap_applied'] in [True,False]
    assert p.summary()['surface_patches_effective']==0


def test_stations_are_persistent_reachable_and_planning_is_not_observation():
    p,c=make_policy();planner=AStarPlanner(p.grid)
    before=json.dumps(p.snapshot())
    class Memory: observations=[]
    for _ in range(3):
        view,path,decision=p.explore(c,Memory(),planner,remaining_distance=50)
        assert decision['station_id']==0
        p.grid.validate_path(path)
        assert view is not None
    assert json.dumps(p.snapshot())==before
    for station in p.stations: assert p.grid.labels[p.grid.require_free(station['position'])]==p.grid.labels[p.grid.require_free(c.position)]
    other,_=make_policy()
    assert [s['position'] for s in p.stations]==[s['position'] for s in other.stations]


def test_only_actual_quality_rgb_updates_effective_coverage():
    p,c=make_policy()
    p.record_arrived(c,np.zeros((120,160,3),np.uint8))
    assert p.summary()['surface_patches_seen']>0
    assert p.summary()['surface_patches_effective']==0
    rgb=np.random.default_rng(3).integers(0,256,(120,160,3),dtype=np.uint8)
    update=p.record_arrived(c,rgb)
    assert update['surface_patches_effective']>0
    assert min(p.first_effective[p.first_effective>=0])==1
    assert update['coverage_update']['effective']
    # A snapshot must not mutate retroactively when more views arrive.
    snapshot=p.snapshot();p.stations[0]['view_first_observed'][0]=100
    assert snapshot['stations'][0]['view_first_observed'][0]!=100


def test_scan_burst_then_move_and_return_pending_directions():
    p,c=make_policy();planner=AStarPlanner(p.grid)
    class Memory: observations=[]
    rgb=np.random.default_rng(3).integers(0,256,(120,160,3),dtype=np.uint8)
    for _ in range(2):
        c,path,decision=p.explore(c,Memory(),planner,50)
        assert decision['station_id']==0 and decision['planner_actual_backend']=='stationary_scan'
        p.record_arrived(c,rgb)
    next_camera,path,decision=p.explore(c,Memory(),planner,50)
    assert decision['station_id']!=0
    assert np.linalg.norm(next_camera.position-c.position)>.5
    assert any(v<0 for v in p.stations[0]['view_first_observed'])
    p.grid.validate_path(path)


def test_station_cap_errors_instead_of_silently_dropping_regions():
    with pytest.raises(MemoryError,match='stations exceeds'):
        make_policy(station_spacing=.5,max_stations=1)
    with pytest.raises(ValueError,match='pitch'):
        make_policy(pitch_degrees=[90])


def test_coverage_first_saves_found_but_keeps_observing(tmp_path):
    c=setup(tmp_path,False)
    c.config['search'].update(mode='coverage_first',max_observations=7,max_path_length=20)
    # This test exercises termination only; it explicitly does NOT validate
    # recognition/identity. Real 8B validation is run separately.
    c.search_mode='coverage_first'
    def scripted_found(candidate,*_):
        if c.observations==2:
            candidate.status='confirmed_TEST_DOUBLE';return True
        return False
    c.memory.confirm=scripted_found
    r=c.run()
    assert r['status']=='FOUND' and r['target_found']
    assert r['observations']==7 and r['stop_reason']=='observation_budget_exhausted'
    assert len(r['confirmed_candidate_ids'])==1
    events=c.artifacts.events
    found=next(i for i,e in enumerate(events) if e['kind']=='target_confirmed')
    assert any(e['kind']=='observation' and e['observation_id']>1 for e in events[found+1:])


def test_coverage_goal_does_not_claim_target_absence_or_found(tmp_path):
    p,camera=make_policy(station_spacing=100.,views_per_visit=1,effective_fraction_goal=.01)
    c=setup(tmp_path,True)
    c.env.camera=camera;c.env.grid=p.grid
    c.env.trajectory=[dict(position=camera.position.tolist(),camera=camera.as_dict(),path_length=0.,action='start')]
    c.policy=p;c.planner=AStarPlanner(p.grid)
    p.record_arrived(camera,np.random.default_rng(5).integers(0,256,(120,160,3),dtype=np.uint8))
    # Reset observation bookkeeping after checking setup; controller must make
    # the real (test-renderer) observation that establishes its own coverage.
    p.first_seen[:]=-1;p.first_effective[:]=-1;p.history=[]
    for s in p.stations: s['view_first_observed']=[-1]*len(s['view_first_observed'])
    r=c.run()
    assert r['status']=='INCOMPLETE' and not r['target_found']
    assert r['coverage_status']=='COMPLETE' and r['stop_reason']=='sampled_coverage_goal_reached'
    assert '不能断言' in r['interpretation']


def test_live_coverage_deltas_preserve_observation_time_and_do_not_render(tmp_path):
    p,c=make_policy()
    v=VisualizationSession(tmp_path,{},dict(export_video=False))
    initial=p.snapshot();v.on_event(dict(kind='coverage_initialized',coverage=initial))
    rgb=np.random.default_rng(3).integers(0,256,(120,160,3),dtype=np.uint8)
    v.capture(rgb,c,'EXPLORE',0)
    report=p.record_arrived(c,rgb)
    v.on_event(dict(kind='observation',observation_id=0,detections=[],coverage=report))
    v.capture(rgb,c,'EXPLORE',0)
    assert v.frames[0]['coverage_step']==v.frames[1]['coverage_step']==0
    assert v.state['coverage_map']['first_effective']==p.first_effective.tolist()
    assert all(x==-1 for x in initial['first_effective'])
    assert v.state['coverage_summary']['surface_patches_effective']>0


def test_global_verification_budget_preserves_spatial_exploration(tmp_path):
    c=setup(tmp_path,False);c.search_mode='coverage_first'
    c.config['search'].update(mode='coverage_first',max_verification_observations=1,max_observations=12,max_path_length=20)
    r=c.run()
    assert r['error'] is None and r['status']=='INCOMPLETE'
    assert r['verification_observations']<=1
    assert r['observations']==12
    assert r['path_length']>1.5


def test_visibility_work_and_cache_have_explicit_resource_guards():
    with pytest.raises(MemoryError,match='cache'):
        make_policy(max_visibility_cache_mb=.0001)
    with pytest.raises(MemoryError,match='ray work'):
        make_policy(max_visibility_ray_steps=1)


def test_multilevel_stations_are_not_merged_by_horizontal_spacing():
    base,_=make_policy()
    multi,_=make_policy(height_offsets=[0,-.6,.6])
    assert len(multi.stations)>len(base.stations)
    positions=np.array([s['position'] for s in multi.stations])
    assert np.ptp(positions[:,1])>=1.0
    for s in multi.stations: multi.grid.require_free(s['position'])
