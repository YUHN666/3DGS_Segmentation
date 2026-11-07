"""Read-only live viewer, bounded RGB recording, offline replay and MP4.

    Display frames are never sent to perception automatically. The controller
    records only camera poses already reached, then decides which are detector
    keyframes. Offline re-rendering has no detector or search-policy interface.
"""
import copy
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from urllib.parse import urlparse, parse_qs, unquote
import numpy as np
from PIL import Image, ImageDraw
from .artifacts import write_json, jsonable
from .motion import validate_options


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    write_json(tmp,data)
    os.replace(tmp,path)


def make_viewer(root, manifest):
    note=Path(root)/'review_correction.json'
    if note.exists():
        manifest=copy.deepcopy(manifest)
        manifest['state']['review_correction']=json.loads(note.read_text())
    template = Path(__file__).with_name('viewer.html').read_text()
    # Target text is untrusted; do not allow a literal </script> in inline JSON.
    data = json.dumps(manifest,ensure_ascii=False,default=jsonable,allow_nan=False).replace('<','\\u003c')
    (Path(root)/'viewer.html').write_text(template.replace('__MANIFEST_JSON__',data))


class VisualizationSession:
    def __init__(self, root, config, options, mode='online'):
        self.root,self.options,self.mode = Path(root).resolve(),dict(options),mode
        validate_options(self.options)
        self.folder=self.root/'visualization'
        self.frames_dir=self.folder/'frames'
        self.frames_dir.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock()
        self.frames=[]
        self.bytes_written=0
        self.started=time.perf_counter()
        self.state=dict(status='INITIALIZING',phase='准备初始化',search_state='EXPLORE',
                        target=config.get('target',{}),frame_count=0,observations=0,vlm_grounding_calls=0,
                        path_length=0.,complete=False,planned_path=[],events=[],map=None,
                        recording_truncated=False,mode=mode,video=None)
        self.server=None
        self.set_phase('准备初始化')
        self.write_viewer()

    def set_phase(self, phase):
        with self.lock:
            self.state['phase']=phase

    @property
    def can_capture(self):
        return not self.state['recording_truncated']

    def configure_map(self,grid,camera,world_up):
        up=int(np.argmax(np.abs(world_up)))
        axes=[a for a in range(3) if a!=up]
        layer=grid.index(camera.position)[up]
        occupied=np.take(grid.occupied,layer,axis=up)
        pixels=np.where(occupied.T[::-1],58,225).astype(np.uint8)
        Image.fromarray(pixels).save(self.folder/'map.png')
        with self.lock:
            self.state['map']=dict(image='visualization/map.png',axes=axes,up_axis=up,
                                   height=float(camera.position[up]),
                                   lower=grid.lower[axes].tolist(),
                                   upper=(grid.lower+grid.shape*grid.voxel)[axes].tolist(),
                                   description='初始相机高度的障碍切片；规划使用三维几何')

    def on_event(self,event):
        with self.lock:
            kind=event['kind']
            if kind=='progress': self.state['phase']=event['phase']
            elif kind=='coverage_initialized':
                self.state['coverage_map']=copy.deepcopy(event['coverage'])
            elif kind=='grid_scan_initialized':
                self.state['grid_scan']=dict(total=event['scan']['views_total'],completed=0,positions_total=16)
            elif kind=='initialized':
                self.state.update(status='RUNNING',phase='开始搜索')
            elif kind=='state_transition':
                self.state['search_state']=event['state']
                if self.frames: self.frames[-1]['state']=event['state']
            elif kind=='view_decision':
                self.state['planned_path']=event.get('path') or []
            elif kind=='observation':
                coverage=event.get('coverage',{})
                if 'grid_scan' in coverage:
                    self.state['grid_scan']=copy.deepcopy(coverage['grid_scan'])
                self.state['coverage_summary']={k:v for k,v in coverage.items() if k!='coverage_update'}
                self.state['coverage_step']=event['observation_id']
                if 'coverage_map' in self.state:
                    model=self.state['coverage_map'];delta=coverage.get('coverage_update',{})
                    for i in delta.get('seen',[]): model['first_seen'][i]=event['observation_id']
                    for i in delta.get('effective',[]): model['first_effective'][i]=event['observation_id']
                    for i,j in delta.get('station_views',[]): model['stations'][i]['view_first_observed'][j]=event['observation_id']
                    model['summary']=self.state['coverage_summary']
                self.state.update(observations=event['observation_id']+1,
                                  vlm_grounding_calls=event['observation_id']+1,
                                  path_length=event.get('path_length',self.state['path_length']))
                if self.frames:
                    frame=self.frames[-1]
                    frame.update(observation_id=event['observation_id'],detections=event['detections'],
                                 coverage_step=event['observation_id'],
                                 candidate_ids=event.get('candidate_ids',[]))
                    self._append(dict(kind='frame_update',frame=copy.deepcopy(frame)))
            self.state['events'].append({k:v for k,v in event.items()
                                         if k in ['kind','elapsed_seconds','phase','state','reason','observation_id','candidate_id']})
            self.state['events']=self.state['events'][-40:]

    def _append(self,event):
        with (self.folder/'timeline.jsonl').open('a') as stream:
            stream.write(json.dumps(event,ensure_ascii=False,default=jsonable,allow_nan=False)+'\n')

    def capture(self,rgb,camera,state,path_length):
        if not self.can_capture: return None
        if len(self.frames)>=self.options.get('max_frames',10000):
            self._truncate('max_frames')
            return None
        index=len(self.frames)
        relative=f'visualization/frames/{index:06d}.jpg'
        dest=self.root/relative
        Image.fromarray(rgb).save(dest,quality=self.options.get('jpeg_quality',85))
        size=dest.stat().st_size
        if self.bytes_written+size>self.options.get('max_disk_mb',1024)*1024**2:
            dest.unlink()
            self._truncate('max_disk_mb')
            return None
        frame=dict(id=index,image=relative,camera=camera.as_dict(),state=state,
                   coverage_step=self.state.get('coverage_step',-1),
                   motion_seconds=index/float(self.options.get('fps',12)),
                   wall_seconds=time.perf_counter()-self.started,path_length=float(path_length),
                   observation_id=None,detections=[],candidate_ids=[])
        with self.lock:
            self.bytes_written+=size
            self.frames.append(frame)
            self.state.update(frame_count=len(self.frames),path_length=float(path_length))
            self._append(dict(kind='frame',frame=frame))
        return frame

    def _truncate(self,reason):
        with self.lock:
            self.state.update(recording_truncated=True,recording_stop_reason=reason)
        print(f'VISUALIZATION_RECORDING_LIMIT={reason}; search continues',flush=True)

    def manifest(self):
        with self.lock:
            return dict(version=1,options=self.options,state=copy.deepcopy(self.state),frames=copy.deepcopy(self.frames))

    def snapshot(self,after=-1):
        with self.lock:
            return dict(options=self.options,state=copy.deepcopy(self.state),
                        frames=copy.deepcopy(self.frames[max(0,after+1):max(0,after+1)+256]),
                        latest=copy.deepcopy(self.frames[-1]) if self.frames else None)

    def write_viewer(self):
        manifest=self.manifest()
        atomic_json(self.folder/'manifest.json',manifest)
        make_viewer(self.root,manifest)

    def finish(self,report):
        with self.lock:
            self.state.update(status=report['status'],search_state=report['status'],
                              observations=report.get('observations',0),vlm_grounding_calls=report.get('vlm_grounding_calls',0),
                              stop_reason=report.get('stop_reason'),complete=True,
                              phase='搜索完成，准备回放',path_length=report.get('path_length',0))
            if self.frames:
                self.frames[-1]['state']=report['status']
                self._append(dict(kind='frame_update',frame=copy.deepcopy(self.frames[-1])))
        self.write_viewer()
        if self.options.get('export_video',True) and self.frames:
            self.set_phase('正在导出 MP4')
            try:
                export_video(self.root,self.manifest())
                self.state['video']='search.mp4'
            except Exception as exc:
                self.state['video_error']=repr(exc)
                write_json(self.folder/'video_error.json',dict(error=repr(exc)))
                print(f'VIDEO_EXPORT_ERROR={exc}',flush=True)
        self.set_phase('回放已就绪' if self.frames else '运行已结束，未生成画面')
        self.write_viewer()
        return dict(viewer='viewer.html',video=self.state['video'],frame_count=len(self.frames),
                    recording_truncated=self.state['recording_truncated'],mode=self.mode,
                    video_error=self.state.get('video_error'),fps=self.options.get('fps',12),
                    preview_frames_are_detector_keyframes=False)


def read_manifest(root):
    value=json.loads((Path(root)/'visualization/manifest.json').read_text())
    note=Path(root)/'review_correction.json'
    if note.exists(): value['state']['review_correction']=json.loads(note.read_text())
    return value


def export_video(root,manifest=None,fps=None):
    root=Path(root).resolve()
    manifest=manifest or read_manifest(root)
    frames=manifest['frames']
    if not frames: raise ValueError('No recorded frames to export')
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg: raise RuntimeError('ffmpeg not found; install ffmpeg, then use export-video')
    rate=float(fps or manifest['options'].get('fps',12))
    if not 0<rate<=60: raise ValueError('Video FPS must be in (0,60]')
    width,height=Image.open(root/frames[0]['image']).size
    canvas_width=width+(width%2)
    canvas_height=height+44+(height%2)
    temp=root/'search.tmp.mp4'
    log=root/'visualization/ffmpeg.log'
    command=[ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24',
             '-video_size',f'{canvas_width}x{canvas_height}','-framerate',str(rate),'-i','pipe:0','-an',
             '-c:v','libx264','-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)]
    with log.open('w') as stderr:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=stderr)
        try:
            for frame in frames:
                with Image.open(root/frame['image']) as source:
                    rgb=source.convert('RGB')
                if rgb.size!=(width,height): raise ValueError('Video frame resolutions differ')
                canvas=Image.new('RGB',(canvas_width,canvas_height),(15,23,42))
                canvas.paste(rgb,(0,44))
                draw=ImageDraw.Draw(canvas)
                keyframe=frame.get('observation_id')
                caption=f"{frame['state']} | frame {frame['id']} | t={frame['id']/rate:.2f}s | distance={frame['path_length']:.2f}"
                draw.text((12,8),caption,fill='white')
                draw.text((12,25),'RGB display frame' if keyframe is None else f'Detector keyframe {keyframe}',fill='#67e8f9')
                for d in frame.get('detections',[]):
                    x1,y1,x2,y2=d['box']
                    status=d.get('review_status','proposal')
                    color='#4ade80' if status.startswith('confirmed') else '#94a3b8' if status.startswith('rejected') else '#fbbf24'
                    draw.rectangle((x1,y1+44,x2,y2+44),outline=color,width=2)
                    score_text='VLM' if d.get('score') is None else f"{d['score']:.2f}"
                    draw.text((x1,max(44,y1+31)),f"{d['label']} {score_text} [{status}]",fill=color)
                process.stdin.write(canvas.tobytes())
            process.stdin.close()
            result=process.wait(timeout=120)
            if result: raise RuntimeError(f'ffmpeg exit {result}: {log.read_text()[-2000:]}')
        except BaseException:
            if process.poll() is None: process.kill()
            process.wait()
            raise
    os.replace(temp,root/'search.mp4')
    return root/'search.mp4'


class ViewerHandler(SimpleHTTPRequestHandler):
    def __init__(self,*args,root,provider,**kwargs):
        self.root=Path(root).resolve()
        self.provider=provider
        super().__init__(*args,directory=str(self.root),**kwargs)

    def log_message(self,format,*args):
        pass

    def end_headers(self):
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        super().end_headers()

    def do_GET(self):
        url=urlparse(self.path)
        if url.path=='/api/state':
            try: after=int(parse_qs(url.query).get('after',['-1'])[0])
            except ValueError:
                self.send_error(400,'Invalid frame cursor');return
            self._json(self.provider.snapshot(after));return
        if url.path=='/api/manifest':
            self._json(self.provider.manifest());return
        relative=unquote(url.path).lstrip('/') or 'viewer.html'
        path=(self.root/relative).resolve()
        if not path.is_relative_to(self.root) or path.is_dir():
            self.send_error(403,'Path not available');return
        if not path.is_file(): self.send_error(404);return
        # Support HTML5 MP4 seeking while keeping every route read-only.
        size=path.stat().st_size
        start,end=0,size-1
        request_range=self.headers.get('Range')
        if request_range:
            match=re.fullmatch(r'bytes=(\d*)-(\d*)',request_range)
            if not match or not any(match.groups()): self.send_error(416);return
            left,right=match.groups()
            if left:
                start=int(left);end=min(size-1,int(right)) if right else size-1
            else: start=max(0,size-int(right))
            if start>=size or end<start: self.send_error(416);return
        self.send_response(206 if request_range else 200)
        self.send_header('Content-Type',mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('Accept-Ranges','bytes')
        if request_range: self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length',str(end-start+1))
        self.end_headers()
        if self.command=='HEAD': return
        try:
            with path.open('rb') as stream:
                stream.seek(start)
                remaining=end-start+1
                while remaining>0:
                    block=stream.read(min(65536,remaining))
                    if not block: break
                    self.wfile.write(block);remaining-=len(block)
        except (BrokenPipeError,ConnectionResetError): pass

    def _json(self,value):
        data=json.dumps(value,ensure_ascii=False,default=jsonable,allow_nan=False).encode()
        self.send_response(200)
        self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(data)))
        self.end_headers()
        if self.command=='HEAD': return
        try: self.wfile.write(data)
        except (BrokenPipeError,ConnectionResetError): pass

    def do_HEAD(self):
        self.do_GET()


class SavedProvider:
    def __init__(self,root): self.data=read_manifest(root)
    def manifest(self): return self.data
    def snapshot(self,after=-1):
        frames=self.data['frames']
        return dict(options=self.data['options'],state=self.data['state'],
                    frames=frames[max(0,after+1):max(0,after+1)+256],latest=frames[-1] if frames else None)


class LiveServer:
    def __init__(self,root,provider,host='127.0.0.1',port=8765):
        self.httpd=ThreadingHTTPServer((host,port),partial(ViewerHandler,root=root,provider=provider))
        self.httpd.daemon_threads=True
        self.thread=threading.Thread(target=self.httpd.serve_forever,name='gs-search-viewer',daemon=True)
        self.thread.start()
        self.url=f'http://{host}:{self.httpd.server_port}/'
        print(f'LIVE_URL={self.url}',flush=True)

    def wait(self):
        print('查看服务保持运行；Ctrl+C 退出。搜索产物已保存在磁盘。',flush=True)
        try:
            while self.thread.is_alive(): self.thread.join(.5)
        except KeyboardInterrupt: pass
        finally: self.close()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)
