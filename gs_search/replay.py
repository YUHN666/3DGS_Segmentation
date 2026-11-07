"""Replay saved evidence or re-render the executed camera trajectory.

No detector/VLM/search policy is instantiated by this module. Interpolated
positions remain on the saved executed polyline and are collision checked.
"""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import time
import uuid
import numpy as np
from PIL import Image
from .artifacts import write_json
from .camera import Camera
from .motion import between
from .visualization import VisualizationSession, LiveServer, make_viewer, read_manifest


def saved_events(root):
    p=Path(root)/'decisions.jsonl'
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def ensure_viewer(root):
    root=Path(root).resolve()
    if (root/'visualization/manifest.json').exists():
        make_viewer(root,read_manifest(root))
        return
    config=json.loads((root/'config.json').read_text())
    report=json.loads((root/'report.json').read_text())
    session=VisualizationSession(root,config,dict(fps=2,export_video=False),mode='keyframes_only')
    for event in saved_events(root):
        if event['kind'] in ['coverage_initialized','grid_scan_initialized']: session.on_event(event)
        if event['kind']!='observation': continue
        with Image.open(root/event['rgb']) as source: rgb=np.asarray(source.convert('RGB'))
        session.capture(rgb,Camera.from_dict(event['camera']),event.get('state','UNKNOWN'),event.get('path_length',0))
        session.on_event(event)
    session.finish(report)


def render_trajectory(source,options,live=False,host='127.0.0.1',port=8765):
    source=Path(source).resolve()
    config=json.loads((source/'config.json').read_text())
    original=json.loads((source/'report.json').read_text())
    trajectory=json.loads((source/'trajectory.json').read_text())
    if not trajectory or any(not isinstance(x,dict) or 'camera' not in x for x in trajectory):
        raise ValueError('Full replay requires a search trajectory containing camera poses; xyz-only plan results are insufficient')
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    output=source/'replays'/f'{stamp}_{uuid.uuid4().hex[:6]}'
    output.mkdir(parents=True)
    session=VisualizationSession(output,config,options,mode='trajectory_rerender')
    server=LiveServer(output,session,host,port) if live else None
    print(f'REPLAY_DIR={output}',flush=True)
    try:
        session.set_phase('加载原场景与轨迹')
        from .scene import load_scene
        from .navigation import OccupancyMap
        from .render import make_renderer
        scene=load_scene(config['scene'])
        grid=OccupancyMap(scene.geometry,config['navigation'])
        # Validate the original executed path before generating new frames.
        cameras=[Camera.from_dict(x['camera']) for x in trajectory]
        grid.validate_path([c.position for c in cameras])
        session.configure_map(grid,cameras[0],config['camera'].get('world_up',[0,-1,0]))
        session.set_phase('加载渲染内核；不调用检测模型')
        renderer=make_renderer(scene,config['renderer'])
        observations={}
        last_observation=None
        for event in saved_events(source):
            if event['kind'] in ['coverage_initialized','grid_scan_initialized']: session.on_event(event)
            elif event['kind']=='observation':
                observations[event['observation_id']]=dict(event)
                last_observation=event['observation_id']
            elif event['kind']=='state_transition' and last_observation is not None:
                observations[last_observation]['state_after']=event['state']
        current=cameras[0]
        state='EXPLORE'
        distance=0.
        started=time.perf_counter()
        session.state.update(status='RUNNING',phase='沿保存轨迹补渲染')
        session.capture(renderer.render(current),current,state,distance)
        for item,end in zip(trajectory[1:],cameras[1:]):
            if not session.can_capture: break
            changed=not (np.allclose(current.c2w,end.c2w,atol=1e-10) and np.allclose(current.K,end.K,atol=1e-10))
            if changed:
                for step in between(current,end,options):
                    if not session.can_capture: break
                    tick=time.perf_counter()
                    if not grid.segment_free(current.position,step.camera.position):
                        raise RuntimeError('Saved replay segment failed continuous collision validation')
                    distance+=float(np.linalg.norm(step.camera.position-current.position))
                    current=step.camera
                    session.capture(renderer.render(current),current,state,distance)
                    if options.get('realtime',False):
                        time.sleep(max(0,step.duration-(time.perf_counter()-tick)))
            if item.get('action')=='observe' and item.get('observation_id') in observations:
                event=observations[item['observation_id']]
                # Repeated detector observations can share a pose. Keep each
                # real keyframe rather than overwrite its predecessor's ID.
                if session.frames and session.frames[-1]['observation_id'] is not None:
                    session.capture(renderer.render(current),current,state,distance)
                session.on_event(event)
                state=event.get('state_after',event.get('state',state))
                session.on_event(dict(kind='state_transition',state=state))
        replay_report=dict(status=original['status'] if session.can_capture else 'INCOMPLETE',
                           stop_reason='saved_trajectory_replayed' if session.can_capture else 'recording_limit',
                           source_run=str(source),source_status=original['status'],
                           observations=len({f['observation_id'] for f in session.frames if f['observation_id'] is not None}),
                           path_length=distance,render_count=renderer.calls,vlm_grounding_calls=0,vlm_calls=0,
                           original_decisions_modified=False,replay_seconds=time.perf_counter()-started)
        replay_report['visualization']=session.finish(replay_report)
        write_json(output/'replay_info.json',replay_report)
        print(f'REPLAY_VIEWER={output/"viewer.html"}',flush=True)
        print(f'REPLAY_VIDEO={output/"search.mp4"}',flush=True)
        return output,server
    except BaseException as exc:
        import traceback
        write_json(output/'replay_error.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        session.finish(dict(status='INCOMPLETE',stop_reason='replay_error',path_length=0,observations=0))
        if server: server.close()
        raise
