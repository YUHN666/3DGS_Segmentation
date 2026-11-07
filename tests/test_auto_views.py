"""Geometry-only camera preparation, synthetic scenes; no model accuracy claims."""
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from gs_search.auto_views import generate_layout,prepare_scene,validate_binding,inherit_captured_geometry
from gs_search.camera import Camera
from gs_search.cli import configure_scene
from gs_search.grid_scan import expand_grid_cameras
from gs_search.navigation import OccupancyMap,PlanningError
from gs_search.scene import Geometry
from gs_search.offline import run_offline,offline_cameras


def room_geometry(shift=(0,0,0),scale=1,multiple=False):
    sites=[]
    for offset in ([0,5] if multiple else [0]):
        for a in np.arange(0,4.01,.25):
            for b in np.arange(0,3.01,.25):
                sites.extend([[offset,b,a],[offset+4,b,a],[offset+a,b,0],[offset+a,b,4]])
        for x in np.arange(0,4.01,.25):
            for z in np.arange(0,4.01,.25): sites.extend([[offset+x,0,z],[offset+x,3,z]])
    means=np.asarray(sites,float)*scale+shift;n=len(means)
    return Geometry(means,np.full((n,3),.09*scale),np.tile([1.,0,0,0],(n,1)),np.ones(n))


def configs():
    return dict(width=400,height=300,fov_degrees=75,world_up=[0,-1,0]),dict(
        bounds=[[-1,-1,-1],[10,4,5]],voxel_size=.16,camera_radius=.08,margin=.04,
        gaussian_sigma=2,opacity_threshold=.02,max_cells=1000000)


def test_128_views_are_collision_free_deterministic_and_geometry_only():
    geometry=room_geometry(multiple=True);camera,nav=configs();original=copy.deepcopy(nav)
    layout,adapted,meta,_=generate_layout(geometry,camera,nav)
    again,_,_,_=generate_layout(geometry,camera,nav)
    assert layout==again and nav==original
    assert len(set(meta['selected_components']))>=2
    assert meta['semantic_inputs_used'] is False and not meta['preview_RGB_used_for_selection']
    grid=OccupancyMap(geometry,adapted);groups=expand_grid_cameras(layout)
    assert len(groups)==16 and sum(map(len,groups))==128
    positions=np.array([g[0].position for g in groups])
    distance=np.linalg.norm(positions[:,None]-positions[None,:],axis=2)+np.eye(16)*100
    assert distance.min()>=meta['min_station_spacing']-1e-8
    for group in groups:
        for view in group: grid.require_free(view.position);Camera.from_dict(view.as_dict())
    assert list(inspect.signature(generate_layout).parameters)==['geometry','camera_config','navigation_config','options']


def test_translated_scaled_scene_and_z_up_supported():
    camera,nav=configs();g=room_geometry(shift=(20,30,40),scale=10)
    layout,adapted,meta,_=generate_layout(g,camera,nav)
    assert meta['distance_parameter_scale']==pytest.approx(10)
    assert adapted['camera_radius']==pytest.approx(.8)
    positions=np.array([s['position'] for s in layout['camera_positions']])
    assert (positions[:,0]>20).all() and (positions[:,2]>40).all()
    rotation=np.array([[1,0,0],[0,0,1],[0,-1,0.]])
    base=room_geometry();base.means=base.means@rotation.T
    camera['world_up']=[0,0,1]
    layout,_,meta,_=generate_layout(base,camera,nav)
    assert layout['up']==[0,0,1] and abs(np.dot(layout['up'],layout['forward']))<1e-8
    assert len(expand_grid_cameras(layout))==16


def test_no_safe_space_and_oversize_grid_fail_without_dropping_obstacles():
    camera,nav=configs();g=room_geometry();g.scales[:]=100
    with pytest.raises(PlanningError,match='No collision-free'): generate_layout(g,camera,nav)
    g=room_geometry();g.means[-1]=[10000,0,0]
    with pytest.raises(MemoryError,match='max_cells'): generate_layout(g,camera,nav)
    with pytest.raises(ValueError): generate_layout(room_geometry(),camera,nav,dict(height_fractions=[1.5]))


def test_saved_binding_and_scale_are_stable_on_reprepare(tmp_path):
    camera,nav=configs();path=tmp_path/'source.ply';path.write_bytes(b'TEST GEOMETRY FINGERPRINT')
    geometry=room_geometry(scale=2)
    scene=SimpleNamespace(geometry=geometry,metadata={'backend':'TEST_GEOMETRY'})
    c=dict(scene=dict(path=str(path)),camera=camera,navigation=nav,renderer=dict(near=.02,far=60),
           localization=dict(near=.02,far=60,min_depth_band=.05),auto_views=dict(mode='auto'))
    original=copy.deepcopy(c)
    prepared,meta=prepare_scene(scene,c,tmp_path/'first')
    second,_=prepare_scene(scene,prepared,tmp_path/'second')
    assert c==original and prepared['navigation']==second['navigation']
    assert prepared['renderer']==second['renderer']
    layout=json.loads((tmp_path/'first/layout.json').read_text());validate_binding(layout,c['scene'])
    assert len(offline_cameras(prepared))==128
    path.write_bytes(b'CHANGED TEST GEOMETRY')
    with pytest.raises(ValueError,match='different/modified'): validate_binding(layout,c['scene'])


def test_scene_override_never_uses_stale_fixed_cameras():
    c=dict(scene=dict(path='/old.ply'),camera=dict(world_up=[0,-1,0]))
    result=configure_scene(SimpleNamespace(scene='/new.ply',camera_mode=None,world_up=None),c)
    assert result['auto_views']['mode']=='auto'
    with pytest.raises(ValueError): configure_scene(SimpleNamespace(scene='/other.ply',camera_mode='fixed',world_up=None),c)


def test_reused_images_keep_capture_scale_without_copying_target_or_device():
    current=dict(target=dict(description='NEW'),camera=dict(world_up=[0,-1,0]),
                 renderer=dict(near=.02,far=60,device='cuda:1'),navigation=dict(voxel_size=.16),
                 localization=dict(far=60),auto_views=dict(mode='auto'))
    source=dict(target=dict(description='OLD'),camera=dict(world_up=[0,0,1]),
                navigation=dict(voxel_size=1.6,bounds=[[0,0,0],[100,100,100]]),
                renderer=dict(near=.2,far=600,device='cuda:0'),localization=dict(far=600),
                auto_views=dict(reference_distances={'navigation':{'voxel_size':.16}}))
    result=inherit_captured_geometry(current,source)
    assert result['target']==current['target'] and result['renderer']['device']=='cuda:1'
    assert result['renderer']['far']==result['localization']['far']==600
    assert result['camera']['world_up']==[0,0,1] and result['navigation']['voxel_size']==1.6
    assert result['auto_views']['reference_distances']['navigation']['voxel_size']==.16


def test_prepare_only_acquires_frames_without_any_semantic_model(tmp_path,monkeypatch):
    monkeypatch.setattr('gs_search.artifacts.diagnostics',lambda:{'test_double':True})
    path=tmp_path/'geometry.ply';path.write_bytes(b'TEST GEOMETRY FINGERPRINT')
    geometry=room_geometry();camera,nav=configs();camera.update(width=32,height=24)
    scene=SimpleNamespace(geometry=geometry,metadata={'backend':'TEST_GEOMETRY'})
    config=dict(scene=dict(path=str(path)),camera=camera,navigation=nav,renderer=dict(backend='gsplat',max_batch=2),
        output_root=str(tmp_path/'runs'),target=dict(description='TEST TARGET',category='chair'),
        vlm=dict(mode='disabled'),auto_views=dict(mode='auto'),offline={},localization=dict(mode='rendered_depth_after_vlm'))
    def forbidden(*_): raise AssertionError('Preparation must not construct semantic models')
    class Renderer:
        calls=0
        def __init__(self,*_): pass
        def render_batch_rgbd(self,cameras):
            self.calls+=len(cameras)
            return [dict(rgb=np.zeros((24,32,3),np.uint8),expected_z=np.ones((24,32)),alpha=np.ones((24,32))) for _ in cameras]
    monkeypatch.setattr('gs_search.scene.load_scene',lambda _:scene)
    monkeypatch.setattr('gs_search.render_rgbd.OfflineRGBDRenderer',Renderer)
    monkeypatch.setattr('gs_search.perception.QwenVerifier',forbidden)
    root,report=run_offline(config,prepare_only=True)
    assert report['status']=='PREPARED' and report['processing_complete']
    assert report['observations']==report['render_count']==128 and report['vlm_calls']==report['sam2_calls']==0
    assert report['grounding_status']=='NOT_RUN' and not report['target_found']
    assert (root/'auto_cameras/layout.json').is_file() and (root/'requested_config.json').is_file()
