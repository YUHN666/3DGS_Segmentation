import argparse
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from PIL import Image
from .artifacts import RunArtifacts, diagnostics, write_json
from .camera import Camera, look_at
from .scene import load_scene
from .navigation import OccupancyMap, AStarPlanner
from .configuration import current_config


def load_config(path):
    p = Path(path).resolve()
    c = current_config(json.loads(p.read_text()))
    # Paths resolve relative to project root (parent of configs), never cwd.
    # Use the config location, so an installed wheel also supports a checkout's
    # configs/ and nested configs/local/ files.
    root = next((parent.parent for parent in p.parents if parent.name == 'configs'), p.parent)
    for section, key in [("scene", "path"), ("vlm", "model_path"),
                         ("splatplan", "repo"), ("camera", "poses_file"), ("segmentation", "checkpoint")]:
        value = c.get(section, {}).get(key)
        if value and not Path(value).is_absolute():
            c[section][key] = str((root/value).resolve())
    value=c.get('policy',{}).get('grid128',{}).get('positions_file')
    if value and not Path(value).is_absolute(): c['policy']['grid128']['positions_file']=str((root/value).resolve())
    if not Path(c["output_root"]).is_absolute():
        c["output_root"] = str((root/c["output_root"]).resolve())
    # Optional machine-local settings; the public templates remain portable.
    if os.environ.get('GS_SEARCH_VLM_DEVICE') and 'vlm' in c:
        c['vlm']['device'] = os.environ['GS_SEARCH_VLM_DEVICE']
    if os.environ.get('GS_SEARCH_SAM2_CHECKPOINT') and 'segmentation' in c:
        c['segmentation']['checkpoint'] = str(Path(os.environ['GS_SEARCH_SAM2_CHECKPOINT']).expanduser().resolve())
    return c


def configure_scene(args,config):
    previous=Path(config['scene']['path']).resolve()
    scene=getattr(args,'scene',None)
    mode=getattr(args,'camera_mode',None)
    up=getattr(args,'world_up',None)
    if scene:
        path=Path(scene).resolve()
        if mode=='fixed' and path!=previous:
            raise ValueError('A different scene needs automatic cameras; use --camera-mode auto or a matching prepared config')
        config['scene']['path']=str(path)
        config.setdefault('auto_views',{})['mode']='auto'
    if up is not None:
        if mode=='fixed': raise ValueError('--world-up requires automatic camera preparation')
        config['camera']['world_up']=up
        config.setdefault('auto_views',{})['mode']='auto'
    if mode: config.setdefault('auto_views',{})['mode']=mode
    return config


def initial_camera(c, index=None):
    conf = c["camera"]
    w, h = conf["width"], conf["height"]
    if index is not None or conf.get("initial_pose_index") is not None:
        obj = json.loads(Path(conf["poses_file"]).read_text())
        poses = obj if isinstance(obj, list) else obj["cameras"]
        d = poses[index if index is not None else conf["initial_pose_index"]]
        pose = np.eye(4)
        if "camera_to_world" in d:
            pose = np.array(d["camera_to_world"], float)
            K = np.array(d["K"], float)
        else:  # Graphdeco cameras.json: rotation is camera-to-world.
            pose[:3, :3], pose[:3, 3] = d["rotation"], d["position"]
            K = np.array([[d["fx"], 0, d["width"]/2], [0, d["fy"], d["height"]/2], [0, 0, 1.]])
        pose[:3, 3] *= c["scene"].get("world_scale", 1)
        K[0] *= w/d["width"]
        K[1] *= h/d["height"]
        camera = Camera(pose, K, w, h)
        if index is None and conf.get("start") is not None:
            camera = camera.moved(conf["start"])
        if conf.get("fov_degrees") is not None and not conf.get("use_source_intrinsics", False):
            camera = with_fov(camera, conf["fov_degrees"])
        return camera
    focal = .5*w/np.tan(np.radians(conf.get("fov_degrees", 75))/2)
    K = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1.]])
    return look_at(conf["start"], conf.get("direction", [0, 0, 1]), conf["world_up"], K, w, h)


def with_fov(camera, degrees):
    if not np.isfinite(degrees) or not 5 <= degrees <= 150:
        raise ValueError("Horizontal FOV must be within [5,150] degrees")
    K = camera.K.copy()
    K[0,0] = K[1,1] = .5*camera.width/np.tan(np.radians(degrees)/2)
    return Camera(camera.c2w, K, camera.width, camera.height)


def make_planner(scene, grid, c):
    backend = c["navigation"].get("backend", "occupancy_astar_3d")
    if backend == "occupancy_astar_3d":
        return AStarPlanner(grid)
    if backend == "splatplan_checked_seed":
        from .splatplan_adapter import SplatPlanAdapter
        primary=SplatPlanAdapter(scene.geometry,grid,c['splatplan'])
        if c['splatplan'].get('fallback_backend')=='occupancy_astar_3d':
            from .planner_fallback import ExplicitPlannerFallback
            return ExplicitPlannerFallback(primary,grid)
        return primary
    raise ValueError(f"Unknown planner backend {backend}")


def discover(roots):
    # Explicit roots only; no home/server-wide scan. Skip label-derived PLYs.
    result = []
    for root in roots:
        p = Path(root).resolve()
        if not p.is_dir():
            continue
        for pattern in ["**/point_cloud.ply", "**/original_rgb_compact.ply"]:
            for f in p.glob(pattern):
                with f.open("rb") as stream:
                    header = stream.read(6000).split(b"end_header")[0].decode(errors="replace")
                result.append(dict(path=str(f), bytes=f.stat().st_size,
                                   gaussian_ply="scale_0" in header and "opacity" in header))
    return sorted(result, key=lambda d:d["bytes"])


def parser():
    p = argparse.ArgumentParser(description="Headless known-geometry / RGB-only 3DGS active search")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("diagnose")
    d.add_argument("--output", default="artifacts/diagnostics.json")
    d.add_argument("--scene-root", action="append", default=[])
    for name in ["render", "plan", "search"]:
        q = sub.add_parser(name)
        q.add_argument("--config", default="configs/search.json")
        q.add_argument("--scene")
        q.add_argument("--planner", choices=["occupancy_astar_3d", "splatplan_checked_seed"])
        q.add_argument("--start", type=float, nargs=3)
        q.add_argument("--fov-degrees", type=float)
        q.add_argument("--snap-start", action="store_true", help="Explicitly opt in to start snapping; saved in decisions")
        if name == "render":
            q.add_argument("--camera-index", type=int)
            q.add_argument("--position", type=float, nargs=3)
            q.add_argument("--direction", type=float, nargs=3)
        if name == "plan":
            q.add_argument("--goal", type=float, nargs=3, required=True)
        if name == "search":
            q.add_argument("--target")
            q.add_argument("--category")
            q.add_argument("--max-observations", type=int)
            q.add_argument("--max-path-length", type=float)
            q.add_argument("--max-seconds", type=float)
            q.add_argument('--exploration-policy',choices=['sampled','coverage','grid128'],help='Sampled exploration, coverage stations, or a fixed 16x8 camera schedule')
            q.add_argument('--search-mode',choices=['search_first','coverage_first'],help='Stop on target, or save target and continue coverage')
            q.add_argument("--live", action="store_true", help="Live browser viewer, motion recording and MP4")
            q.add_argument("--record-video", action="store_true", help="Record motion and export MP4 without serving")
            q.add_argument("--live-port", type=int, default=8765)
            q.add_argument("--live-host", default="127.0.0.1")
            q.add_argument("--fps", type=float)
            q.add_argument("--camera-speed", type=float, help="Scene units per second")
            q.add_argument("--angular-speed", type=float, help="Degrees per second")
            q.add_argument("--no-realtime", action="store_true", help="Render as fast as possible; playback retains configured FPS")
    r = sub.add_parser("replay")
    r.add_argument("run_dir")
    r.add_argument("--render-trajectory", action="store_true", help="Re-render saved actual motion with no detector calls")
    r.add_argument("--serve", action="store_true")
    r.add_argument("--host", default="127.0.0.1")
    r.add_argument("--port", type=int, default=8765)
    r.add_argument("--fps", type=float, default=12)
    r.add_argument("--camera-speed", type=float, default=.35)
    r.add_argument("--angular-speed", type=float, default=45)
    r.add_argument("--realtime", action="store_true")
    s = sub.add_parser("serve")
    s.add_argument("run_dir")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    v = sub.add_parser("export-video")
    v.add_argument("run_dir")
    v.add_argument("--fps",type=float)
    o=sub.add_parser('offline',help='Acquire/reuse 128 RGB views, directly ground targets with VLM, review and mark Gaussian PLY')
    o.add_argument('--config',default='configs/offline.json')
    o.add_argument('--scene',help='Gaussian PLY; automatically prepare cameras for this scene')
    o.add_argument('--camera-mode',choices=['auto','fixed'],help='Automatically prepare geometry-based cameras, or use the configured layout')
    o.add_argument('--world-up',type=float,nargs=3,help='Scene up direction, e.g. 0 0 1; default follows the template convention')
    o.add_argument('--prepare-only',action='store_true',help='Prepare cameras and acquire 128 RGB/geometry frames, without loading VLM or SAM2')
    o.add_argument('--target')
    o.add_argument('--category',help='English category for independent review')
    o.add_argument('--reuse-run',help='Reuse RGB/geometry from 128 saved views; VLM regenerates boxes unless --reuse-grounding is set')
    o.add_argument('--reuse-grounding',action='store_true',help='Reuse same-target VLM raw responses from --reuse-run; repair invalid responses within budget')
    g=sub.add_parser('project-ply',help='Mark accepted saved masks on original Gaussian PLY; no model inference')
    g.add_argument('run_dir',help='Offline result containing candidates, masks, geometry and cameras')
    a=sub.add_parser('prepare-scene',help='Geometry-only automatic 128-camera configuration, no rendering or model inference')
    a.add_argument('--scene',required=True)
    a.add_argument('--config',default='configs/offline.json')
    a.add_argument('--world-up',type=float,nargs=3)
    a.add_argument('--output-root',help='Optional destination for the preparation result directory')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command=='prepare-scene':
        from .auto_views import prepare_scene
        c=configure_scene(args,load_config(args.config))
        if args.output_root: c['output_root']=str(Path(args.output_root).resolve())
        run=RunArtifacts(c['output_root'],c,'prepare_scene')
        print(f'RESULT_DIR={run.path}',flush=True)
        write_json(run.path/'requested_config.json',c)
        try:
            c,meta=prepare_scene(load_scene(c['scene']),c,run.path/'auto_cameras')
            write_json(run.path/'config.json',c)
            run.finish(dict(status='CAMERAS_PREPARED',camera_preparation=meta,observations=0,
                            render_count=0,vlm_calls=0,navigation_executed=False,error=None),[])
            print(f"PREPARED_CONFIG={run.path/'config.json'}",flush=True)
            print(f"REPORT={run.path/'auto_cameras/index.html'}",flush=True)
            return 0
        except Exception as exc:
            error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
            run.finish(dict(status='INCOMPLETE',error=error,observations=0,render_count=0,vlm_calls=0),[])
            print(error['traceback'],file=sys.stderr);return 1
    if args.command=='project-ply':
        from .gaussian_projection import project_saved_run
        result=project_saved_run(args.run_dir)
        print(f"PLY_DIR={Path(args.run_dir).resolve()/result['directory']}")
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    if args.command=='offline':
        from .offline import run_offline
        c=configure_scene(args,load_config(args.config))
        if args.target: c['target']['description']=args.target
        if args.category: c['target']['category']=args.category
        c['vlm']['category_prompt']=c['target']['category']
        _,report=run_offline(c,args.reuse_run,args.reuse_grounding,args.prepare_only)
        return 1 if report['error'] or report.get('gaussian_projection_error') else 0
    if args.command == "diagnose":
        d = diagnostics()
        if args.scene_root:
            d["scene_candidates"] = discover(args.scene_root)
        Path(args.output).resolve().parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, d)
        print(json.dumps(d, ensure_ascii=False, indent=2))
        return 0
    if args.command in ["replay","serve","export-video"]:
        from .replay import ensure_viewer, render_trajectory
        from .visualization import LiveServer, SavedProvider, export_video
        root=Path(args.run_dir).resolve()
        if args.command=="replay" and args.render_trajectory:
            options=dict(fps=args.fps,camera_speed=args.camera_speed,angular_speed=args.angular_speed,
                         realtime=args.realtime,export_video=True,max_frames=10000,max_disk_mb=1024)
            output,server=render_trajectory(root,options,live=args.serve,host=args.host,port=args.port)
            if server: server.wait()
            return 0
        ensure_viewer(root)
        if args.command=="export-video":
            print(export_video(root,fps=args.fps))
            return 0
        if (root/'report.json').exists():
            run=RunArtifacts.__new__(RunArtifacts)
            run.path=root
            run.html_report()
        print(root/'viewer.html')
        if args.command=="serve" or args.serve:
            LiveServer(root,SavedProvider(root),args.host,args.port).wait()
        return 0
    c = load_config(args.config)
    if args.scene:
        c["scene"]["path"] = str(Path(args.scene).resolve())
    if args.planner:
        c["navigation"]["backend"] = args.planner
    if args.start:
        c["camera"]["start"] = args.start
    if args.fov_degrees is not None:
        c["camera"]["fov_degrees"] = args.fov_degrees
        c["camera"]["use_source_intrinsics"] = False
    if args.command == "search":
        if args.exploration_policy is not None: c['policy']['strategy']=args.exploration_policy
        if args.search_mode is not None: c['search']['mode']=args.search_mode
        visualization=c.setdefault('visualization',{})
        if args.live or args.record_video: visualization['enabled']=True
        if args.live: visualization['realtime']=not args.no_realtime
        if args.no_realtime: visualization['realtime']=False
        for flag in ['fps','camera_speed','angular_speed']:
            if getattr(args,flag) is not None: visualization[flag]=getattr(args,flag)
        for flag, key in [("max_observations", "max_observations"), ("max_path_length", "max_path_length"), ("max_seconds", "max_seconds")]:
            if getattr(args, flag) is not None:
                c["search"][key] = getattr(args, flag)
        if args.target:
            c["target"]["description"] = args.target
        if args.category:
            c["target"]["category"] = args.category
        c["target"]["require_full_description"] = (c["target"]["description"].strip().lower().strip(".") !=
                                                     c["target"]["category"].strip().lower().strip("."))
        c['vlm']['category_prompt']=c['target']['category']
        c['vlm']['review_input']='context_panel'
    run = RunArtifacts(c["output_root"], c, args.command)
    print(f"RESULT_DIR={run.path}", flush=True)
    trajectory = []
    server = None
    try:
        init_time = time.perf_counter()
        if args.command=='search' and c['visualization'].get('enabled',False):
            from .visualization import VisualizationSession, LiveServer
            run.visualization=VisualizationSession(run.path,c,c['visualization'])
            if args.live:
                server=LiveServer(run.path,run.visualization,args.live_host,args.live_port)
        run.progress('加载 3DGS 场景')
        scene = load_scene(c["scene"])
        write_json(run.path/"scene.json", scene.metadata)
        camera = initial_camera(c, getattr(args, "camera_index", None))
        if args.command == "render":
            if args.position:
                camera = camera.moved(args.position)
            if args.direction:
                camera = look_at(camera.position, args.direction, c["camera"]["world_up"], camera.K, camera.width, camera.height)
            from .render import make_renderer
            renderer = make_renderer(scene, c["renderer"])
            t = time.perf_counter()
            rgb = renderer.render(camera)
            run.observation(0, rgb, camera, [], [], dict(state="RENDER"))
            report = dict(status="RENDERED", renderer=renderer.backend, render_count=1,
                          rgb_mean=float(rgb.mean()), rgb_std=float(rgb.std()),
                          nonblack_fraction=float((rgb.max(2)>5).mean()), render_seconds=time.perf_counter()-t,
                          camera=camera.as_dict())
            if report["rgb_std"] < 2 or report["nonblack_fraction"] < .02:
                raise RuntimeError("Render is blank/degenerate; inspect saved RGB and camera")
            run.finish(report, [camera.as_dict()])
        else:
            if c["scene"].get("navigation_enabled") is False:
                raise ValueError("Render-only scene config: supply and validate navigation bounds/start before enabling navigation")
            run.progress('构建三维几何导航地图')
            grid = OccupancyMap(scene.geometry, c["navigation"])
            write_json(run.path/"geometry.json", grid.metadata)
            if args.snap_start:
                original = camera.position
                camera = camera.moved(grid.snap(original))
                run.event("explicit_start_snap", original=original, replacement=camera.position)
                c["camera"]["start"] = camera.position.tolist()
                write_json(run.path/"config.json", c)
            grid.require_free(camera.position, "start")
            if run.visualization is not None:
                run.visualization.configure_map(grid,camera,c['camera'].get('world_up',[0,-1,0]))
            run.progress('初始化路径规划器')
            planner = make_planner(scene, grid, c)
            if args.command == "plan":
                path = planner.plan(camera.position, args.goal)
                write_json(run.path/"path.json", path)
                write_json(run.path/"planner_details.json", planner.last_metadata)
                run.event("planned_path", path=path, start=camera.position, goal=args.goal)
                run.finish(dict(status="PLANNED", planner=planner.backend, continuous_collision_validated=grid.validate_path(path),
                                path_length=float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
                                start=camera.position, goal=args.goal), path.tolist())
            else:
                from .render import make_renderer
                from .perception import QwenVerifier
                from .vlm_grounding import VLMGrounder
                from .policy import ViewPolicy
                from .controller import CameraEnvironment, SearchController
                run.progress('加载 gsplat 渲染内核；首次或工具链变化时会编译')
                renderer = make_renderer(scene, c["renderer"])
                if c['vlm'].get('mode') not in ['local','service']:
                    raise ValueError('Search requires an enabled local or configured service VLM')
                model_name=Path(c['vlm'].get('model_path') or c['vlm'].get('model','configured-model')).name
                run.progress(f"加载视觉语言复核模型 {model_name} [{c['vlm'].get('dtype','bfloat16')}]"
                             if c['vlm'].get('mode')!='disabled' else '类别搜索；VLM 已关闭')
                verifier = QwenVerifier(c["vlm"])
                grounder=VLMGrounder(verifier,c.get('grounding',{}).get('max_candidates_per_image',12))
                if verifier.mode=='local':
                    info=verifier.model_info
                    print(f"[vlm] parameters={info.get('parameters'):,} dtype={info['dtype']} "
                          f"quantization={info['quantization']} device={info['device']} "
                          f"weights_GiB={info.get('parameter_memory_bytes',0)/1024**3:.2f}",flush=True)
                strategy=c['policy'].get('strategy','sampled')
                if strategy=='coverage':
                    from .coverage import CoveragePolicy
                    run.progress('构建固定观察站点与独立几何可见性地图')
                    policy=CoveragePolicy(scene.geometry,grid,c['policy'],c['camera'])
                    policy.initialize(camera)
                elif strategy=='sampled':
                    policy = ViewPolicy(scene.geometry, grid, c["policy"], c["camera"])
                elif strategy=='grid128':
                    from .grid_scan import FixedGridScan
                    run.progress('加载固定 16×8 视角；检查全部位置与起点连通性')
                    policy=FixedGridScan(scene.geometry,grid,c['policy'],c['camera'])
                    policy.initialize(camera)
                else: raise ValueError(f'Unknown policy strategy {strategy}')
                env = CameraEnvironment(camera, grid, renderer)
                run.event("initialized", seconds=time.perf_counter()-init_time, planner=planner.backend)
                report = SearchController(env, planner, grounder, verifier, policy, run, c).run()
                print(json.dumps({k:report[k] for k in ["status", "stop_reason", "path_length", "observations", "vlm_grounding_calls"]}, ensure_ascii=False))
                if report["error"]:
                    print(report["error"]["traceback"], file=sys.stderr)
                    if server: server.close()
                    return 1
        print(f"REPORT={run.path/'index.html'}", flush=True)
        if run.visualization is not None:
            print(f"VIEWER={run.path/'viewer.html'}",flush=True)
            if run.visualization.state.get('video'):
                print(f"VIDEO={run.path/'search.mp4'}",flush=True)
        if server:
            if args.command=='search':
                # The read-only replay server needs saved RGB/JSON, not models.
                # Release model/scene tensors before waiting for the browser.
                del renderer,grounder,verifier,planner,env,policy
                import gc
                import torch
                gc.collect()
                for device in range(torch.cuda.device_count()):
                    with torch.cuda.device(device): torch.cuda.empty_cache()
                run.progress('回放服务就绪；搜索模型显存已释放')
                run.visualization.write_viewer()
            server.wait()
        return 0
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        if "planner" in locals():
            write_json(run.path/"planner_details.json", planner.last_metadata)
        run.event("error", **error)
        run.finish(dict(status="INCOMPLETE", stop_reason="initialization_or_command_error", error=error,
                        observations=0, render_count=getattr(locals().get("renderer"), "calls", 0),
                        vlm_grounding_calls=getattr(locals().get("grounder"), "calls", 0), vlm_calls=0,
                        path_length=0, timing_seconds={"command": time.perf_counter()-run.started}), trajectory)
        print(error["traceback"], file=sys.stderr)
        if server: server.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
