from datetime import datetime
from pathlib import Path
import html
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import uuid
import numpy as np
from PIL import Image, ImageDraw


def jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=jsonable, allow_nan=False)+"\n")


def diagnostics():
    import torch
    packages = {}
    for p in ["torch", "torchvision", "gsplat", "transformers", "huggingface-hub", "numpy", "scipy", "numba",
              "plyfile", "Pillow", "opencv-python-headless", "safetensors", "accelerate", "bitsandbytes", "SAM-2", "plotly", "pytest", "open3d", "clarabel", "cvxpy", "cvxopt", "dijkstra3d", "unfoldNd", "polytope"]:
        try:
            packages[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            packages[p] = None
    def command(cmd):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            return dict(returncode=r.returncode, stdout=r.stdout.strip(), stderr=r.stderr.strip())
        except Exception as e:
            return dict(error=repr(e))
    repos = {}
    for name in ["splatnav", "vlfm"]:
        repos[name] = command(["git", "-C", str(Path(__file__).resolve().parents[1]/"third_party"/name), "log", "-1", "--format=%H %cI %s"])
    sam=importlib.util.find_spec('sam2')
    if sam and sam.origin:
        repos['sam2']=command(['git','-C',str(Path(sam.origin).resolve().parent.parent),'log','-1','--format=%H %cI %s'])
    available_models = {}
    for name in ["qwen", "qwen8b", "sam2"]:
        revision_file = Path(__file__).resolve().parents[1]/"artifacts"/f"{name}_model_revision.json"
        if revision_file.exists():
            available_models[name] = json.loads(revision_file.read_text())
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"), cuda_available=torch.cuda.is_available(),
                CUDA_DEVICE_ORDER=os.environ.get("CUDA_DEVICE_ORDER"),
                torch_cuda=torch.version.cuda,
                gpu=[dict(logical_id=i, name=torch.cuda.get_device_name(i),
                          free_total_bytes=torch.cuda.mem_get_info(i)) for i in range(torch.cuda.device_count())],
                packages=packages, third_party=repos, available_model_snapshots=available_models,
                compiler_environment=dict(CXX=os.environ.get('CXX'),CC=os.environ.get('CC')),
                cxx=command([os.environ.get('CXX','c++'),'--version']),
                nvidia_smi=command(["nvidia-smi", "-i", "2,5", "--query-gpu=index,name,uuid,driver_version,memory.free", "--format=csv,noheader"]),
                nvcc=command(["nvcc", "--version"]), gcc=command(["gcc", "--version"]),
                cmake=command(["cmake", "--version"]))


class RunArtifacts:
    def __init__(self, root, config, command):
        # UTC+8 user-facing folder timestamp, independent of host timezone.
        from datetime import timezone, timedelta
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
        self.path = Path(root).resolve()/f"{stamp}_{command}_{uuid.uuid4().hex[:6]}"
        self.path.mkdir(parents=True)
        for name in ["rgb", "annotated", "crops"]:
            (self.path/name).mkdir()
        self.started = time.perf_counter()
        self.events = []
        self.visualization = None
        write_json(self.path/"config.json", config)
        write_json(self.path/"environment.json", diagnostics())

    def event(self, kind, **data):
        obj = dict(elapsed_seconds=time.perf_counter()-self.started, kind=kind, **data)
        self.events.append(obj)
        with (self.path/"decisions.jsonl").open("a") as f:
            f.write(json.dumps(obj, ensure_ascii=False, default=jsonable, allow_nan=False)+"\n")
        if self.visualization is not None:
            self.visualization.on_event(obj)

    def progress(self, phase):
        print(f"[progress] {phase}", flush=True)
        self.event("progress", phase=phase)

    def observation(self, obs_id, rgb, camera, detections, candidate_ids, extra):
        name = f"{obs_id:04d}.png"
        Image.fromarray(rgb).save(self.path/"rgb"/name)
        annotated = Image.fromarray(rgb)
        draw = ImageDraw.Draw(annotated)
        for i, d in enumerate(detections):
            color=('#4ade80' if d.review_status.startswith('confirmed') else
                   '#94a3b8' if d.review_status.startswith('rejected') else '#fbbf24')
            draw.rectangle(d.box, outline=color, width=2)
            score_text='VLM' if d.score is None else f'{d.score:.3f}'
            draw.text((d.box[0], max(0, d.box[1]-12)), f"{d.label} {score_text} [{d.review_status}]", fill=color)
            Image.fromarray(rgb).crop(tuple(d.box)).save(self.path/"crops"/f"{obs_id:04d}_{i:02d}.png")
        annotated.save(self.path/"annotated"/name)
        self.event("observation", observation_id=obs_id, rgb=f"rgb/{name}", annotated=f"annotated/{name}",
                   camera=camera.as_dict(), detections=[d.as_dict() for d in detections],
                   candidate_ids=candidate_ids, **extra)

    def finish(self, report, trajectory):
        if self.visualization is not None:
            report['visualization'] = self.visualization.finish(report)
        report["elapsed_seconds"] = time.perf_counter()-self.started
        write_json(self.path/"report.json", report)
        write_json(self.path/"trajectory.json", trajectory)
        self.html_report()
        return self.path

    def html_report(self):
        report = json.loads((self.path/"report.json").read_text())
        if report.get('mode')=='offline128':
            from .offline import html_report
            html_report(self.path,report)
            return
        events = [json.loads(line) for line in (self.path/"decisions.jsonl").read_text().splitlines()] if (self.path/"decisions.jsonl").exists() else []
        cards = []
        for e in events:
            if e["kind"] == "observation":
                cards.append('<article><h3>Observation '+str(e["observation_id"])+
                             '</h3><a href="'+e["rgb"]+'"><img src="'+e["annotated"]+'"></a><pre>'+
                             html.escape(json.dumps({k:v for k,v in e.items() if k != "camera"}, ensure_ascii=False, indent=2))+
                             '</pre></article>')
        replay_link = '<a href="viewer.html">实时 / 连续回放</a> · ' if (self.path/'viewer.html').exists() else ''
        if (self.path/'coverage.json').exists():
            replay_link += '<a href="coverage.json">表面覆盖与站点记录</a> · '
        if (self.path/'grid_scan.json').exists():
            replay_link += '<a href="grid_scan.json">固定 128 视角计划与进度</a> · '
        if (self.path/'contact_sheet.jpg').exists():
            replay_link += '<a href="contact_sheet.jpg">观察图像总览</a> · '
        correction=''
        if (self.path/'review_correction.json').exists():
            note=json.loads((self.path/'review_correction.json').read_text())
            correction=('<aside style="background:#ffe4e6;color:#881337;padding:16px;border:2px solid #e11d48">'
                        '历史 FOUND 已被人工反馈否定，本次不计为成功：'+html.escape(note['reason'])+
                        ' <a href="review_correction.json">复核记录</a></aside>')
        text = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>3DGS active search</title>'
                '<style>body{font:16px sans-serif;background:#eee;margin:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere}'
                'main{display:flex;flex-wrap:wrap;gap:16px}article{background:white;padding:12px;width:440px}'
                'img{max-width:100%}</style>'+correction+'<h1>3DGS 主动搜索实际观察报告</h1><p>'
                + replay_link + '<a href="report.json">JSON report</a> · <a href="trajectory.json">Trajectory</a> · '
                '<a href="decisions.jsonl">Decision log</a></p><pre>'+html.escape(json.dumps(report, ensure_ascii=False, indent=2))+
                '</pre><main>'+''.join(cards)+'</main></html>')
        (self.path/"index.html").write_text(text)
