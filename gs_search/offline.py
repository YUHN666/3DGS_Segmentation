"""Offline 128-view acquisition -> direct RGB VLM review -> opt-in RGB-D localization.

This deliberately separate mode implements the user's revised depth allowance.
No motion preview, navigation trajectory, association gate before VLM, or depth
input to a detector/VLM. Every semantic result remains a model judgment.
"""
import gc
import hashlib
import html
import json
from pathlib import Path
import shutil
import time
import traceback
from types import SimpleNamespace
import numpy as np
from PIL import Image,ImageDraw
from .artifacts import RunArtifacts,write_json
from .camera import Camera
from .grid_scan import expand_grid_cameras
from .memory import review_panel
from .perception import Detection
from .configuration import current_config,target_identity


def rgb_panel(rgb,detection):
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
        raise TypeError('Review requires uint8 RGB')
    h,w=rgb.shape[:2]
    return review_panel(SimpleNamespace(rgb=rgb,detection=detection,camera=SimpleNamespace(width=w,height=h)),1)


def review_detection(rgb,detection,target_text,verifier):
    """No geometry/depth, feature matching, border gate, or multiview requirement."""
    image,_=review_input(rgb,detection,verifier)
    return verifier.verify([image],target_text)


def review_input(rgb,detection,verifier):
    if getattr(verifier,'config',{}).get('review_input')!='focus_crop':
        return rgb_panel(rgb,detection),dict(format='context_panel')
    box=np.asarray(detection.box,float);h,w=rgb.shape[:2]
    margin=max(16,.25*max(box[2:]-box[:2]))
    region=[max(0,int(np.floor(box[0]-margin))),max(0,int(np.floor(box[1]-margin))),
            min(w,int(np.ceil(box[2]+margin))),min(h,int(np.ceil(box[3]+margin)))]
    crop=Image.fromarray(rgb).crop(tuple(region));scale=768/max(crop.size)
    crop=crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))),Image.Resampling.LANCZOS)
    return np.asarray(crop),dict(format='unmarked_focus_crop',region_original_pixels=region,
                                resizing='Lanczos interpolation, not new rendered detail')


def release_gpu():
    gc.collect()
    import torch
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def offline_cameras(config):
    layout=json.loads(Path(config['policy']['grid128']['positions_file']).read_text())
    from .auto_views import validate_binding
    validate_binding(layout,config['scene'])
    groups=expand_grid_cameras(layout)
    order=[0,1,2,3,7,6,5,4,8,9,10,11,15,14,13,12]
    if config['policy']['grid128'].get('position_order')=='reference': order=list(range(16))
    return [dict(name=f'{layout["camera_positions"][i]["name"]}_y{j:02d}',**groups[i][j].as_dict()) for i in order for j in range(8)]


def source_observations(root,config):
    root=Path(root).resolve();source=json.loads((root/'config.json').read_text())
    if Path(source['scene']['path']).resolve()!=Path(config['scene']['path']).resolve() or source['scene'].get('world_scale',1)!=config['scene'].get('world_scale',1):
        raise ValueError('Reuse scene path/world scale differs; refusing geometry association')
    events=[json.loads(s) for s in (root/'decisions.jsonl').read_text().splitlines()]
    observations=[e for e in events if e['kind']=='observation']
    if len(observations)!=128: raise ValueError('Reuse requires exactly 128 saved observations')
    return root,observations


def run_offline(config,reuse_run=None,reuse_grounding=False,prepare_only=False):
    from .perception import QwenVerifier
    from .render_rgbd import OfflineRGBDRenderer
    from .scene import load_scene
    config=current_config(config)
    config['vlm']['category_prompt']=config['target']['category']
    run=RunArtifacts(config['output_root'],config,'offline128')
    print(f'RESULT_DIR={run.path}',flush=True)
    for folder in ['geometry','reviews','masks','pointclouds','grounding']: (run.path/folder).mkdir()
    rows=[];candidates=[];renderer=None;verifier=None;segmenter=None
    render_count=0;error=None;complete=False
    grounding_records=[];grounding_calls=0;review_calls=0;proposal_complete=True
    grounding_cache_reads=0;reused_grounding=0
    timings={};started=time.perf_counter();disk_limit=config.get('offline',{}).get('max_disk_mb',2048)*1024**2
    mode=config.get('localization',{}).get('mode','disabled')
    camera_preparation=None
    try:
        if reuse_grounding and not reuse_run:
            raise ValueError('--reuse-grounding requires --reuse-run')
        if not prepare_only and config['vlm'].get('mode') not in ['local','service']:
            raise ValueError('VLM grounding requires an enabled local or configured service VLM')
        if mode not in ['disabled','rendered_depth_after_vlm']:
            raise ValueError('localization.mode must be disabled/rendered_depth_after_vlm')
        if mode!='disabled' and config.get('renderer',{}).get('backend')!='gsplat':
            raise ValueError('Offline expected-Z localization requires gsplat')
        t=time.perf_counter()
        if reuse_run:
            source,observations=source_observations(reuse_run,config)
            camera_preparation=dict(mode='reused_observations',source=str(source),views=128)
            source_config=json.loads((source/'config.json').read_text())
            if config.get('auto_views',{}).get('mode')=='auto':
                from .auto_views import inherit_captured_geometry,validate_binding
                binding_path=source/'auto_cameras/layout.json'
                if binding_path.exists(): validate_binding(json.loads(binding_path.read_text()),config['scene'])
                write_json(run.path/'requested_config.json',config)
                config=inherit_captured_geometry(config,source_config)
                write_json(run.path/'config.json',config)
                camera_preparation['geometry_parameters']='inherited from captured source configuration'
            if reuse_grounding:
                if target_identity(source_config)!=target_identity(config):
                    raise ValueError('Grounding cache target/category differs')
                for key in ['mode','model_path','model','url','dtype','quantization']:
                    if source_config['vlm'].get(key)!=config['vlm'].get(key):
                        raise ValueError(f'Grounding cache model {key} differs')
            for i,e in enumerate(observations):
                if e['observation_id']!=i: raise ValueError('Reuse observations must be contiguous 0..127')
                image_path=(source/e['rgb']).resolve()
                if not image_path.is_relative_to(source): raise ValueError('Reuse image path escapes source directory')
                if e['camera']['width']*e['camera']['height']>config['renderer'].get('max_pixels',1_500_000):
                    raise MemoryError('Reuse RGB exceeds configured pixel limit')
                with Image.open(image_path) as im:
                    if im.size!=(e['camera']['width'],e['camera']['height']): raise ValueError('Reuse image dimensions differ')
                shutil.copyfile(image_path,run.path/'rgb'/f'{i:04d}.png')
                detections=[]
                # Cached RGB-D is bound to exactly this saved camera and RGB.
                # Copy available geometry; older RGB-only runs are still rendered
                # only after semantic acceptance, with RGB parity validation.
                geometry_path=source/'geometry'/f'{i:04d}.npz'
                if mode!='disabled' and geometry_path.is_file():
                    with np.load(geometry_path,allow_pickle=False) as geometry:
                        shape=(e['camera']['height'],e['camera']['width'])
                        if any(geometry[k].shape!=shape for k in ['expected_z','alpha']): raise ValueError('Cached geometry dimensions differ')
                    shutil.copyfile(geometry_path,run.path/'geometry'/geometry_path.name)
                rows.append(dict(id=i,name=f'reused_{i:03d}',camera=Camera.from_dict(e['camera']),detections=detections))
            run.event('reused_rgb',source=str(source),views=128,category=config['target']['category'])
        else:
            run.progress('离线直接渲染 128 张 RGB 与对应几何；不生成运动帧')
            scene=load_scene(config['scene']);write_json(run.path/'scene.json',scene.metadata)
            if config.get('auto_views',{}).get('mode','fixed')=='auto':
                from .auto_views import prepare_scene
                run.progress('根据高斯几何自动准备安全相机位置与128视角')
                write_json(run.path/'requested_config.json',config)
                config,camera_preparation=prepare_scene(scene,config,run.path/'auto_cameras')
                write_json(run.path/'config.json',config)
                run.event('automatic_cameras_prepared',details=camera_preparation)
            cameras=offline_cameras(config)
            # Validate positions only; no navigation path is claimed/executed.
            if config.get('offline',{}).get('validate_positions',True):
                from .navigation import OccupancyMap
                grid=OccupancyMap(scene.geometry,config['navigation'])
                for c in cameras: grid.require_free(np.asarray(c['camera_to_world'])[:3,3],'offline camera')
                write_json(run.path/'geometry_map.json',grid.metadata)
                del grid
            renderer=OfflineRGBDRenderer(scene,config['renderer'])
            batch=min(config['renderer'].get('max_batch',2),config.get('offline',{}).get('render_batch',2))
            if batch<1: raise ValueError('render_batch must be >=1')
            cursor=0
            while cursor<len(cameras):
                selected=cameras[cursor:cursor+batch]
                try: outputs=renderer.render_batch_rgbd([Camera.from_dict(c) for c in selected])
                except renderer.torch.OutOfMemoryError:
                    if batch==1: raise
                    batch=max(1,batch//2);release_gpu();run.event('render_batch_reduced',batch=batch);continue
                for c,output in zip(selected,outputs):
                    i=cursor
                    Image.fromarray(output['rgb']).save(run.path/'rgb'/f'{i:04d}.png')
                    if mode!='disabled': np.savez_compressed(run.path/'geometry'/f'{i:04d}.npz',expected_z=output['expected_z'],alpha=output['alpha'])
                    rows.append(dict(id=i,name=c['name'],camera=Camera.from_dict(c),detections=[]))
                    cursor+=1
                if sum(f.stat().st_size for f in run.path.rglob('*') if f.is_file())>disk_limit: raise MemoryError('Offline output exceeds max_disk_mb')
                print(f'[render] {cursor}/128',flush=True)
            render_count=renderer.calls
            del renderer;renderer=None;del scene;release_gpu()
        timings['acquisition']=time.perf_counter()-t
        write_json(run.path/'cameras.json',dict(cameras=[dict(name=row['name'],**row['camera'].as_dict()) for row in rows],mode='offline_independent_views'))
        if prepare_only:
            complete=len(rows)==128
        else:
            from .vlm_grounding import ground_rgb,parse_grounding
            t=time.perf_counter();run.progress('用 Qwen 直接定位 128 张 RGB 中的目标')
            verifier=QwenVerifier(config['vlm'])
            for row in rows:
                if time.perf_counter()-started>config.get('offline',{}).get('max_seconds',1800):
                    proposal_complete=False;run.event('grounding_budget_exhausted',next_observation=row['id']);break
                rgb=np.asarray(Image.open(run.path/'rgb'/f"{row['id']:04d}.png").convert('RGB'))
                maximum=config.get('grounding',{}).get('max_candidates_per_image',12)
                cache=source/'grounding'/f"{row['id']:04d}.json" if reuse_grounding else None
                response=None;cached=False;attempts=[]
                if cache and cache.is_file():
                    previous=json.loads(cache.read_text());grounding_cache_reads+=1
                    response=dict(prompt=previous.get('prompt'),cache_source=str(cache),
                                  **parse_grounding(previous['raw_response'],rgb.shape[1],rgb.shape[0],maximum))
                    cached=response['status']!='INVALID_RESPONSE'
                retries=config.get('grounding',{}).get('invalid_retries',0)
                if not 0<=retries<=2: raise ValueError('grounding.invalid_retries must be 0..2')
                for attempt in range(retries+1):
                    if response is not None and response['status']!='INVALID_RESPONSE': break
                    if (grounding_calls>=config.get('grounding',{}).get('max_calls',128) or
                        time.perf_counter()-started>config.get('offline',{}).get('max_seconds',1800)):
                        proposal_complete=False;break
                    if response is not None:
                        name=f"grounding/{row['id']:04d}_invalid_{attempt}.json"
                        write_json(run.path/name,response);attempts.append(name)
                    before=verifier.calls
                    try:
                        response=ground_rgb(rgb,config['target']['description'],verifier,
                                            min(maximum,3) if response is not None else maximum)
                    finally:
                        grounding_calls+=verifier.calls-before
                if response is None:
                    proposal_complete=False;run.event('grounding_budget_exhausted',next_observation=row['id']);break
                if cached: reused_grounding+=1
                response['previous_invalid_attempts']=attempts
                write_json(run.path/'grounding'/f"{row['id']:04d}.json",response)
                grounding_records.append(dict(observation_id=row['id'],status=response['status'],
                    candidates=len(response['candidates']),invalid=response['invalid'],
                    reused=cached,
                    response=f"grounding/{row['id']:04d}.json"))
                if response['status']=='INVALID_RESPONSE' or response['invalid']: proposal_complete=False
                row['detections']=[Detection(c['box'],None,c['label'],provenance=dict(
                    stage='vlm_grounding',bbox_normalized=c['bbox_normalized'],response=f"grounding/{row['id']:04d}.json"))
                    for c in response['candidates']]
                run.event('offline_grounding',**grounding_records[-1])
                print(f"[ground] {row['id']+1}/128 candidates={len(row['detections'])} vlm_calls={verifier.calls}",flush=True)
            timings['grounding']=time.perf_counter()-t
            # Proposed boxes require independent category/attribute review.
            for row in rows:
                for j,d in enumerate(row['detections']):
                    candidates.append(dict(id=f'candidate_{len(candidates):04d}',observation_id=row['id'],detection_index=j,
                        box=d.box,score=d.score,label=d.label,status='pending_single_view_vlm',semantic_review={},target_xyz=None,
                        localization=dict(status='NOT_ATTEMPTED',target_xyz=None)))
                    candidates[-1]['detection_provenance']=d.provenance
            write_json(run.path/'candidates.json',candidates)
            if candidates:
                t=time.perf_counter();run.progress(f'送入独立类别/属性复核的候选：{len(candidates)} 个')
                if verifier is None: verifier=QwenVerifier(config['vlm'])
                cap=config.get('offline',{}).get('max_vlm_candidates',256)
                for index,c in enumerate(candidates):
                    if index>=cap or time.perf_counter()-started>config.get('offline',{}).get('max_seconds',1800):
                        c['status']='review_skipped_budget';continue
                    row=rows[c['observation_id']];d=row['detections'][c['detection_index']]
                    rgb=np.asarray(Image.open(run.path/'rgb'/f"{row['id']:04d}.png").convert('RGB'))
                    panel,review_meta=review_input(rgb,d,verifier)
                    c['review_input']=review_meta
                    Image.fromarray(panel).save(run.path/'reviews'/f"{c['id']}.png")
                    print(f"[vlm] {index+1}/{len(candidates)} {c['id']} observation={row['id']}",flush=True)
                    before=verifier.calls
                    try:
                        c['semantic_review']=review_detection(rgb,d,config['target']['description'],verifier)
                    finally:
                        review_calls+=verifier.calls-before
                    accepted=c['semantic_review'].get('matches_target') is True
                    decision=c['semantic_review'].get('decision','uncertain')
                    c['status']='accepted_single_view_vlm' if accepted else ('rejected_vlm' if decision=='reject' else 'uncertain_vlm')
                    d.candidate_id=c['id'];d.review_status='confirmed_single_view_vlm' if accepted else c['status']
                    run.event('offline_vlm_review',candidate=c)
                    write_json(run.path/'candidates.json',candidates)
                timings['vlm']=time.perf_counter()-t
            accepted=[c for c in candidates if c['status']=='accepted_single_view_vlm']
            if accepted and mode=='rendered_depth_after_vlm':
                t=time.perf_counter();run.progress(f'仅为 VLM 通过的 {len(accepted)} 个候选分割和反投影')
                from .localization import SAM2BoxSegmenter,backproject_mask,write_colored_cloud
                import cv2
                segmenter=SAM2BoxSegmenter(config['segmentation'])
                for c in accepted:
                    row=rows[c['observation_id']];camera=row['camera']
                    rgb=np.asarray(Image.open(run.path/'rgb'/f"{row['id']:04d}.png").convert('RGB'))
                    geometry_file=run.path/'geometry'/f"{row['id']:04d}.npz"
                    if not geometry_file.exists():
                        # Reused RGB has no depth. Render ONLY the accepted pose,
                        # and require RGB agreement before attaching new geometry.
                        if renderer is None:
                            renderer=OfflineRGBDRenderer(load_scene(config['scene']),config['renderer'])
                        output=renderer.render_batch_rgbd([camera])[0]
                        difference=int(np.abs(output['rgb'].astype(int)-rgb.astype(int)).max())
                        if difference>config.get('localization',{}).get('reuse_rgb_max_error',2):
                            c['localization']=dict(status='UNRESOLVED',reason='reused_rgb_geometry_mismatch',max_rgb_error=difference,target_xyz=None)
                            continue
                        np.savez_compressed(geometry_file,expected_z=output['expected_z'],alpha=output['alpha'])
                    mask,meta=segmenter.segment(rgb,c['box'])
                    # Keep mask inside the queried object box; avoid depth at mask
                    # boundaries where blended splats mix surfaces.
                    x1,y1,x2,y2=np.asarray(c['box'],float)
                    inside=np.zeros(mask.shape,bool);inside[max(0,int(np.floor(y1))):min(mask.shape[0],int(np.ceil(y2))),max(0,int(np.floor(x1))):min(mask.shape[1],int(np.ceil(x2)))]=True
                    mask &=inside
                    erosion=config.get('localization',{}).get('mask_erode',1)
                    if not 0<=erosion<=8: raise ValueError('Invalid mask_erode')
                    if erosion: mask=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8),iterations=erosion).astype(bool)
                    Image.fromarray(mask.astype(np.uint8)*255).save(run.path/'masks'/f"{c['id']}.png")
                    overlay=rgb.copy();overlay[mask]=(.55*overlay[mask]+.45*np.array([0,255,80])).astype(np.uint8)
                    Image.fromarray(overlay).save(run.path/'masks'/f"{c['id']}_overlay.png")
                    with np.load(geometry_file) as geometry:
                        loc,points,pixels=backproject_mask(mask,geometry['expected_z'],geometry['alpha'],camera,config['localization'])
                    c['localization']=dict(loc,segmentation=meta,mask=f"masks/{c['id']}.png")
                    if points is not None:
                        cloud=f"pointclouds/{c['id']}.ply";write_colored_cloud(run.path/cloud,points,rgb[pixels[:,0],pixels[:,1]])
                        c['localization']['pointcloud']=cloud;c['target_xyz']=loc['target_xyz']
                    run.event('offline_localization',candidate_id=c['id'],localization=c['localization'])
                    write_json(run.path/'candidates.json',candidates)
                timings['localization']=time.perf_counter()-t
            complete=proposal_complete and all(c['status']!='review_skipped_budget' for c in candidates)
    except (Exception,KeyboardInterrupt) as exc:
        error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        run.event('error',**error);print(error['traceback'],flush=True)
    if renderer is not None: render_count+=renderer.calls
    # Persist exactly one observation record per discrete RGB, with final labels.
    for row in rows:
        rgb=np.asarray(Image.open(run.path/'rgb'/f"{row['id']:04d}.png").convert('RGB'))
        run.observation(row['id'],rgb,row['camera'],row['detections'],
                        [c['id'] for c in candidates if c['observation_id']==row['id']],dict(state='OFFLINE',path_length=0))
    accepted=[c for c in candidates if c['status']=='accepted_single_view_vlm']
    localized=[c for c in accepted if c['localization'].get('status')=='LOCALIZED']
    report=dict(mode='offline128',status='PREPARED' if prepare_only and complete and error is None else ('FOUND' if accepted else 'INCOMPLETE'),target_found=bool(accepted),
        stop_reason=('offline_pipeline_error' if error else ('rgb_acquisition_only' if prepare_only else 'offline_128_completed')
            if complete else ('offline_grounding_incomplete' if not proposal_complete else 'offline_review_budget_exhausted')),
        preparation_only=prepare_only,camera_preparation=camera_preparation,
        target=config['target'],confirmation_level='single_view_direct_vlm_model_judgment',
        candidates=candidates,accepted_candidate_ids=[c['id'] for c in accepted],
        localized_candidate_ids=[c['id'] for c in localized],observations=len(rows),render_count=render_count,
        reused_rgb_observations=128 if reuse_run else 0,
        vlm_calls=verifier.calls if verifier else 0,vlm_model_info=getattr(verifier,'model_info',None),
        proposal_backend='vlm_grounding',vlm_grounding_calls=grounding_calls,vlm_review_calls=review_calls,
        grounding_status='NOT_RUN' if prepare_only else ('COMPLETE' if proposal_complete and len(grounding_records)==128 else 'PARTIAL'),
        grounding_observations=len(grounding_records),
        grounding_cache_reads=grounding_cache_reads,reused_grounding_observations=reused_grounding,
        score_policy='VLM boxes have no numeric confidence score; category and attributes require review',
        vlm_inference_metrics=getattr(verifier,'metrics',[]),sam2_calls=segmenter.calls if segmenter else 0,
        scan_status='COMPLETE' if len(rows)==128 else 'PARTIAL',
        localization_status=(('COMPLETE' if len(localized)==len(accepted) else 'PARTIAL') if mode!='disabled' else 'NOT_ENABLED') if accepted else 'NOT_APPLICABLE',
        processing_complete=complete and error is None,
        navigation_executed=False,path_length=0.,trajectory_kind='none_offline_independent_cameras',
        localization_policy=mode,depth_definition='alpha-normalized expected camera Z (gsplat RGB+ED)',
        recognition_inputs='RGB images/crops/panels and target/category text only; depth/geometry never provided to Qwen',
        instances_merged=False,timing_seconds=timings,error=error,
        limits=['VLM approval is uncalibrated single-view semantic evidence, not multiview identity confirmation',
                'Rendered depth localization explicitly enabled for this revised offline task',
                '3D estimate is visible-surface median in scene units, not a complete object center',
                'Separate observations are not automatically merged into a single instance'])
    write_json(run.path/'candidates.json',candidates)
    write_json(run.path/'grounding.json',grounding_records)
    if localized and config.get('gaussian_projection',{}).get('enabled',False):
        t=time.perf_counter()
        try:
            from .gaussian_projection import export_gaussian_ply
            report['gaussian_projection']=export_gaussian_ply(run.path,config,candidates)
            run.event('gaussian_projection',result=report['gaussian_projection'])
        except Exception as exc:
            report['gaussian_projection_error']=dict(message=str(exc),traceback=traceback.format_exc())
            report['processing_complete']=False
            run.event('gaussian_projection_error',**report['gaussian_projection_error'])
        timings['gaussian_projection']=time.perf_counter()-t
    if localized and config.get('offline',{}).get('export_scene_html',False):
        try:
            from .localization import export_scene_html
            report['localization_viewer']=export_scene_html(run.path,config,candidates)
        except Exception as exc:
            report['localization_viewer_error']=repr(exc)
    if rows: contact_sheet(run.path,rows)
    run.finish(report,[])
    print(f"REPORT={run.path/'index.html'}",flush=True)
    print(json.dumps(dict(status=report['status'],proposals=len(candidates),vlm_accepted=len(accepted),localized=len(localized),
                          observations=len(rows),render_count=render_count,vlm_calls=report['vlm_calls']),ensure_ascii=False),flush=True)
    return run.path,report


def contact_sheet(root,rows):
    canvas=Image.new('RGB',(8*160,16*140),(20,25,35));draw=ImageDraw.Draw(canvas)
    for i,row in enumerate(rows):
        image=Image.open(root/'annotated'/f"{row['id']:04d}.png").convert('RGB');image.thumbnail((160,120))
        x,y=i%8*160,i//8*140;canvas.paste(image,(x,y+20));draw.text((x+4,y+3),f"{row['id']:03d} {row['name']}",fill='white')
    canvas.save(root/'contact_sheet.jpg',quality=90)


def html_report(root,report):
    root=Path(root);cards=[]
    projection=report.get('gaussian_projection',{})
    projection_link=(f'<p><a href="{html.escape(projection["directory"])}/index.html">'
                     '下载标回原始高斯的完整场景 PLY、红色高亮 PLY 和目标高斯子集</a></p>') if projection.get('directory') else ''
    for c in report.get('candidates',[]):
        loc=c['localization'];cloud=loc.get('pointcloud')
        color='#dcfce7' if c['status']=='accepted_single_view_vlm' else '#fee2e2' if c['status']=='rejected_vlm' else '#fef3c7'
        links=(f'<a href="{html.escape(cloud)}">下载目标表面 PLY</a>' if cloud else '三维位置尚未生成')
        score_text='VLM 直接定位，无数值置信度' if c.get('score') is None else f'检测分数 {c["score"]:.4f}'
        mask_image=(f'<p>用于反投影的掩码：</p><img src="masks/{c["id"]}_overlay.png">'
                    if (root/'masks'/f'{c["id"]}_overlay.png').exists() else '')
        cards.append(f'<article style="background:{color}"><h3>{html.escape(c["id"])} · {html.escape(c["status"])}</h3>'
            f'<a href="rgb/{c["observation_id"]:04d}.png"><img src="reviews/{c["id"]}.png"></a>'
            f'<p>{score_text} · {links}</p>'+mask_image+'<pre>'+html.escape(json.dumps(
                dict(semantic_review=c['semantic_review'],localization=loc),ensure_ascii=False,indent=2))+'</pre></article>')
    summary={k:v for k,v in report.items() if k not in ['candidates','vlm_inference_metrics']}
    text=('<!doctype html><html lang="zh"><meta charset="utf-8"><title>离线目标筛选与三维定位</title>'
          '<style>body{font:16px sans-serif;margin:24px;background:#f1f5f9}main{display:flex;gap:16px;flex-wrap:wrap}'
          'article{width:520px;padding:14px}img{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
          '<h1>128 视角离线筛选与三维定位</h1><p>VLM 判断与三维定位分别报告；绿色表示单图 VLM 接受，不表示多视图确认。</p>'+projection_link+
          ('<p>本次只准备相机和图像，尚未运行目标识别。</p>' if report.get('preparation_only') else '')+
          ('<p><a href="auto_cameras/index.html">自动相机位置与几何检查</a></p>' if (root/'auto_cameras/index.html').exists() else '')+
          '<p><a href="report.json">完整报告</a> · <a href="candidates.json">全部候选与复核</a> · '
          '<a href="contact_sheet.jpg">128 视角总览</a> · <a href="cameras.json">相机参数</a> · '
          +('<a href="localization_3d.html">三维目标位置查看</a>' if (root/'localization_3d.html').exists() else '')+'</p><pre>'+
          html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre><main>'+''.join(cards)+'</main></html>')
    (root/'index.html').write_text(text)
