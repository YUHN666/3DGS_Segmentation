import json
from pathlib import Path
import subprocess
import urllib.request
import urllib.error
import numpy as np
import pytest
from gs_search.camera import Camera
from gs_search.motion import between,along_path,validate_options
from gs_search.visualization import VisualizationSession,LiveServer,read_manifest


def cam(position=(0,0,0)):
    pose=np.eye(4);pose[:3,3]=position
    return Camera(pose,[[80,0,50],[0,80,30],[0,0,1]],100,60)


def test_motion_samples_translation_rotation_and_no_corner_shortcut():
    a=cam();b=cam([1,0,0]);b.c2w[:3,:3]=np.diag([-1,1,-1])
    steps=list(between(a,b,dict(fps=10,camera_speed=.5,angular_speed=90)))
    assert len(steps)==20
    assert np.allclose(steps[-1].camera.c2w,b.c2w)
    assert all(np.linalg.norm(s.camera.position-[s.camera.position[0],0,0])<1e-9 for s in steps)
    turn=list(between(a,Camera(b.c2w.copy(),b.K,100,60).moved([0,0,0]),dict(fps=10,camera_speed=.5,angular_speed=90)))
    assert len(turn)==20 and all(np.allclose(s.camera.position,[0,0,0]) for s in turn)
    route=np.array([[0,0,0],[0,0,1],[1,0,1]],float)
    samples=list(along_path(a,cam([1,0,1]),route,dict(fps=10,camera_speed=1,angular_speed=90)))
    assert any(np.allclose(s.camera.position,[0,0,1]) for s in samples)
    assert all(abs(s.camera.position[0])<1e-9 or abs(s.camera.position[2]-1)<1e-9 for s in samples)
    assert sum(s.arrived for s in samples)==1
    with pytest.raises(ValueError): validate_options(dict(fps=0))


def test_readonly_server_incremental_frames_range_and_path_boundary(tmp_path):
    v=VisualizationSession(tmp_path,{'target':{'description':'</script><script>bad()</script>'}},dict(export_video=False))
    rgb=np.full((60,100,3),120,np.uint8)
    v.capture(rgb,cam(),'EXPLORE',0)
    server=LiveServer(tmp_path,v,port=0)
    try:
        root=server.url
        data=json.load(urllib.request.urlopen(root+'api/state?after=-1'))
        assert len(data['frames'])==1
        assert len(json.load(urllib.request.urlopen(root+'api/state?after=0'))['frames'])==0
        v.on_event(dict(kind='observation',observation_id=0,detections=[dict(box=[10,10,30,30],score=.8,label='chair')],candidate_ids=['c0']))
        patch=json.load(urllib.request.urlopen(root+'api/state?after=0'))
        assert patch['latest']['observation_id']==0 and len(patch['frames'])==0
        request=urllib.request.Request(root+'visualization/frames/000000.jpg',headers={'Range':'bytes=0-9'})
        with urllib.request.urlopen(request) as response:
            assert response.status==206 and len(response.read())==10
        outside=tmp_path.parent/'forbidden.txt';outside.write_text('private')
        (tmp_path/'escape.txt').symlink_to(outside)
        for name in ['escape.txt','%2e%2e/forbidden.txt']:
            with pytest.raises(urllib.error.HTTPError) as exc: urllib.request.urlopen(root+name)
            assert exc.value.code==403
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(urllib.request.Request(root+'api/state',data=b'{}',method='POST'))
        assert exc.value.code==501
        assert '</script><script>bad()' not in (tmp_path/'viewer.html').read_text()
    finally: server.close()


def test_capture_limits_and_actual_h264_export(tmp_path):
    v=VisualizationSession(tmp_path,{},dict(fps=10,max_frames=3,export_video=True))
    for i in range(4): v.capture(np.full((60,100,3),i*60,np.uint8),cam([i*.1,0,0]),'VERIFY',i*.1)
    assert v.state['recording_truncated'] and len(v.frames)==3
    metadata=v.finish(dict(status='INCOMPLETE',observations=0,path_length=.2))
    assert metadata['video_error'] is None
    result=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(tmp_path/'search.mp4')]))
    stream=result['streams'][0]
    assert stream['codec_name']=='h264' and int(stream['nb_frames'])==3
    assert read_manifest(tmp_path)['state']['recording_truncated']


def test_preview_frames_never_add_vlm_grounding_calls(tmp_path):
    from test_controller import setup
    controller=setup(tmp_path,False)
    v=VisualizationSession(controller.artifacts.path,controller.config,
                            dict(fps=12,camera_speed=.35,angular_speed=45,export_video=False))
    controller.artifacts.visualization=v
    controller.visual=v
    result=controller.run()
    assert result['error'] is None
    assert result['observations']==result['vlm_grounding_calls']==3
    assert result['render_count']>result['vlm_grounding_calls']
    assert result['visualization']['frame_count']==result['render_count']
    assert sum(f['observation_id'] is not None for f in v.frames)==3
    controller.env.grid.validate_path([np.asarray(f['camera']['camera_to_world'])[:3,3] for f in v.frames])


def test_legacy_replay_preserves_stationary_observations_without_models(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from gs_search.scene import Geometry
    from gs_search.replay import render_trajectory
    from gs_search.artifacts import write_json
    from test_controller import ScriptedRGB
    source=tmp_path/'original';source.mkdir()
    c=cam([.1,.1,.1])
    config=dict(scene={},navigation=dict(bounds=[[-3,-2,-3],[3,2,3]],voxel_size=.2,camera_radius=.04,margin=.02),
                renderer={},camera=dict(world_up=[0,-1,0]),target=dict(description='chair'))
    write_json(source/'config.json',config)
    write_json(source/'report.json',dict(status='INCOMPLETE',observations=2))
    write_json(source/'trajectory.json',[dict(action='start',camera=c.as_dict()),
                                       *[dict(action='observe',camera=c.as_dict(),observation_id=i) for i in range(2)]])
    events=[dict(kind='observation',observation_id=i,detections=[],camera=c.as_dict(),state='EXPLORE') for i in range(2)]
    (source/'decisions.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    geometry=Geometry(np.empty((0,3)),np.empty((0,3)),np.empty((0,4)),np.empty(0))
    monkeypatch.setattr('gs_search.scene.load_scene',lambda _:SimpleNamespace(geometry=geometry))
    renderer=ScriptedRGB()
    monkeypatch.setattr('gs_search.render.make_renderer',lambda *_:renderer)
    def forbidden(*_): raise AssertionError('Replay must never instantiate a semantic model')
    monkeypatch.setattr('gs_search.vlm_grounding.VLMGrounder.__init__',forbidden)
    monkeypatch.setattr('gs_search.perception.QwenVerifier.__init__',forbidden)
    output,_=render_trajectory(source,dict(fps=12,camera_speed=.35,angular_speed=45,export_video=False))
    data=read_manifest(output)
    assert [f['observation_id'] for f in data['frames']]==[0,1]
    info=json.loads((output/'replay_info.json').read_text())
    assert info['vlm_grounding_calls']==info['vlm_calls']==0
