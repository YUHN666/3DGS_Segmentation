"""Map accepted RGB masks to ORIGINAL Gaussian rows, preserving the 3DGS schema.

Post-recognition geometry only. Labels mean visible target-support Gaussians,
not complete object segmentation, instance IDs, or calibrated probabilities.
"""
from datetime import datetime
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
import traceback
import uuid
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from .artifacts import write_json
from .camera import Camera
from .scene import load_scene


DEFAULTS = dict(depth_abs=.04, depth_rel=.01, min_alpha=.6, min_opacity=.02,
                near=.02, far=60., chunk_size=200000, min_views=1,
                max_export_bytes=1_000_000_000, highlight_rgb=[1., .05, .05])


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''): digest.update(data)
    return digest.hexdigest()


def options(config):
    cfg={**DEFAULTS,**config.get('gaussian_projection',{})}
    for key in ['depth_abs','depth_rel','min_alpha','min_opacity','near','far']:
        if not np.isfinite(cfg[key]) or cfg[key]<0: raise ValueError(f'Invalid projection {key}')
    if cfg['min_alpha']>1 or cfg['min_opacity']>1 or cfg['far']<=cfg['near']:
        raise ValueError('Invalid alpha/opacity/depth limits')
    if not 1<=cfg['chunk_size']<=1_000_000 or not 1<=cfg['min_views']<=65535:
        raise ValueError('Invalid chunk_size/min_views')
    color=np.asarray(cfg['highlight_rgb'])
    if color.shape!=(3,) or not np.isfinite(color).all() or (color<0).any() or (color>1).any():
        raise ValueError('highlight_rgb must have three values in [0,1]')
    return cfg


def select_gaussians(means,opacity,camera,mask,depth,alpha,cfg):
    """Center-in-mask AND rendered-surface-depth agreement; bounded CPU chunks.

    Positive camera Z, pixel floor convention matches pixel centers u+.5/v+.5.
    No Gaussian semantic fields, detector boxes, or predefined target XYZ used.
    """
    shape=(camera.height,camera.width)
    mask=np.asarray(mask,bool);depth=np.asarray(depth);alpha=np.asarray(alpha)
    if mask.shape!=shape or depth.shape!=shape or alpha.shape!=shape:
        raise ValueError('Projection mask/depth/alpha dimensions must match camera')
    valid_pixel=mask&np.isfinite(depth)&np.isfinite(alpha)&(alpha>=cfg['min_alpha'])&(
        depth>cfg['near'])&(depth<cfg['far'])
    matrix=np.linalg.inv(camera.c2w);selected=[]
    for start in range(0,len(means),cfg['chunk_size']):
        xyz=np.asarray(means[start:start+cfg['chunk_size']],np.float64)
        pc=xyz@matrix[:3,:3].T+matrix[:3,3];z=pc[:,2]
        good=np.isfinite(pc).all(1)&(z>cfg['near'])&(z<cfg['far'])&(
            opacity[start:start+len(pc)]>=cfg['min_opacity'])
        indices=np.flatnonzero(good)
        homogeneous=pc[indices]@camera.K.T
        uv=homogeneous[:,:2]/homogeneous[:,2:]
        inside=np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<camera.width)&(
            uv[:,1]>=0)&(uv[:,1]<camera.height)
        indices=indices[inside];uv=np.floor(uv[inside]).astype(np.int64)
        u,v=uv[:,0],uv[:,1];d=depth[v,u]
        hit=valid_pixel[v,u]&(np.abs(z[indices]-d)<=cfg['depth_abs']+cfg['depth_rel']*d)
        selected.append(indices[hit]+start)
    return np.concatenate(selected).astype(np.int64) if selected else np.empty(0,np.int64)


def _inside(root,relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root): raise ValueError('Evidence path escapes run directory')
    return path


def _write_ply(source,data,path,comment,subset=False):
    vertex=PlyElement.describe(data,'vertex',comments=source['vertex'].comments)
    elements=[vertex if element.name=='vertex' else element for element in source.elements]
    # Faces/other elements can refer to original row numbers; omit for a subset.
    if subset: elements=[vertex]
    PlyData(elements,text=False,byte_order=source.byte_order,
            comments=[*source.comments,comment],obj_info=source.obj_info).write(path)


def export_gaussian_ply(root,config,candidates,output_dir=None):
    """Export a new full scene, full highlighted scene, and original-row subset.

    Source PLY is never overwritten. An independent report is saved on failures.
    Multiple same-class observations retain independent index lists; the binary
    union is a target-class selection, never an instance association decision.
    """
    root=Path(root).resolve()
    output=Path(output_dir).resolve() if output_dir else root/'gaussian_projection'/(
        datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();report=dict(status='RUNNING',source_run=str(root),views=[],skipped=[])
    try:
        cfg=options(config);report['parameters']=cfg
        scene=load_scene(config['scene']);source_path=Path(scene.metadata['path'])
        source_hash=sha256(source_path);source=PlyData.read(source_path,mmap=True);raw=source['vertex'].data
        report['source_ply']=str(source_path);report['source_sha256']=source_hash
        report['source_gaussians']=len(raw);report['invalid_geometry_excluded']=scene.metadata['invalid_removed']
        report['world_scale_for_projection']=scene.metadata['world_scale']
        report['output_coordinates']='Original PLY coordinates and raw parameter activations, unchanged'
        # Bound output allocation/disk; no silent source subsampling.
        estimate=len(raw)*(raw.dtype.itemsize+3)*3+source_path.stat().st_size
        if estimate>cfg['max_export_bytes'] or estimate>shutil.disk_usage(output).free:
            raise MemoryError(f'PLY export estimated bytes {estimate} exceeds configured/free disk budget')
        report['estimated_max_export_bytes']=estimate
        cameras=json.loads((root/'cameras.json').read_text())['cameras']
        vote_count=np.zeros(len(raw),np.uint16);per_view={};index_lists={}
        for candidate in candidates:
            loc=candidate.get('localization',{});review=candidate.get('semantic_review',{})
            if (candidate.get('status')!='accepted_single_view_vlm' or review.get('matches_target') is not True
                    or loc.get('status')!='LOCALIZED'):
                report['skipped'].append(dict(candidate_id=candidate['id'],reason='requires_VLM_accept_and_localized_mask'))
                continue
            obs=int(candidate['observation_id'])
            if not 0<=obs<len(cameras): raise ValueError('Observation index outside saved cameras')
            camera=Camera.from_dict(cameras[obs]);mask_path=_inside(root,loc['mask'])
            mask=np.asarray(Image.open(mask_path).convert('L'))>0
            depth_path=root/'geometry'/f'{obs:04d}.npz'
            with np.load(depth_path,allow_pickle=False) as geometry:
                indices=select_gaussians(scene.geometry.means,scene.geometry.opacity,camera,mask,
                                         geometry['expected_z'],geometry['alpha'],cfg)
            original_rows=scene.source_indices[indices]
            key=candidate['id']
            if not key.replace('_','').isalnum() or key in index_lists: raise ValueError('Invalid/duplicate candidate ID')
            index_lists[key]=original_rows
            per_view.setdefault(obs,[]).append(original_rows)
            report['views'].append(dict(candidate_id=key,observation_id=obs,selected_gaussians=len(original_rows),
                mask=str(mask_path.relative_to(root)),mask_sha256=sha256(mask_path),depth_sha256=sha256(depth_path),
                camera=camera.as_dict(),source_indices_key=key))
        for lists in per_view.values(): vote_count[np.unique(np.concatenate(lists))]+=1
        selected=vote_count>=cfg['min_views'];indices=np.flatnonzero(selected)
        columns=[('search_target','u1'),('search_view_count','u2')]
        if set(raw.dtype.names)&{n for n,_ in columns}: raise ValueError('PLY already contains search export fields')
        labeled=np.empty(len(raw),dtype=raw.dtype.descr+columns)
        for name in raw.dtype.names: labeled[name]=raw[name]
        labeled['search_target']=selected;labeled['search_view_count']=vote_count
        _write_ply(source,labeled,output/'scene_target_labels.ply','Target labels from accepted RGB masks + rendered-depth visibility')
        _write_ply(source,labeled[selected],output/'target_gaussians.ply','Selected ORIGINAL Gaussian rows; not XYZRGB surface points',subset=True)
        np.save(output/'target_gaussian_indices.npy',indices)
        np.savez_compressed(output/'candidate_gaussian_indices.npz',**index_lists)
        # Selected geometry, opacity and rotation are untouched. Zero only their
        # higher SH coefficients so the highlight has a fixed color in every view.
        for axis in range(3): labeled[f'f_dc_{axis}'][selected]=(cfg['highlight_rgb'][axis]-.5)/.28209479177387814
        for name in raw.dtype.names:
            if name.startswith('f_rest_'): labeled[name][selected]=0
        _write_ply(source,labeled,output/'scene_target_highlighted.ply','Selected Gaussian SH replaced by highlight; geometry unchanged')
        unchanged=sha256(source_path)==source_hash
        if not unchanged: raise RuntimeError('Source PLY changed during projection')
        report.update(status='EXPORTED' if len(indices) else 'EMPTY',selected_gaussians=len(indices),
            selected_fraction=len(indices)/max(1,len(raw)),source_unchanged=unchanged,
            label_semantics=dict(search_target='0/1 accepted target-class visible Gaussian support, not instance ID',
                                 search_view_count='Number of distinct accepted observation IDs supporting this row'),
            instances_merged=False,method='original Gaussian center inside accepted SAM2 mask + camera-Z depth agreement',
            versions={name:importlib.metadata.version(name) for name in ['numpy','plyfile','scipy']},
            limits=['Only visible supported Gaussian centers are selected; hidden surfaces are not inferred',
                    'Expected rendered depth mixes splats; this is approximate Gaussian membership',
                    'VLM acceptance and SAM2 masks can be wrong; no manual ground-truth accuracy claim'],
            files={name:dict(path=name,bytes=(output/name).stat().st_size,sha256=sha256(output/name)) for name in
                   ['scene_target_labels.ply','scene_target_highlighted.ply','target_gaussians.ply',
                    'target_gaussian_indices.npy','candidate_gaussian_indices.npz']})
        write_json(output/'config.json',config)
    except Exception as exc:
        report.update(status='ERROR',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise
    finally:
        report['timing_seconds']=time.perf_counter()-started
        write_json(output/'projection_report.json',report)
        links=''.join(f'<li><a href="{html.escape(name)}">{html.escape(name)}</a></li>' for name in report.get('files',{}))
        (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>原始 3DGS PLY 目标标记</h1>'
            '<p>labels：完整原场景加标签；highlighted：完整原场景中目标为红色；target_gaussians：目标原始高斯子集。'
            '保留尺度/旋转/不透明度/SH 格式。原文件不覆盖；标签不是实例 ID。</p><ul>'+links+'</ul><pre>'+
            html.escape(json.dumps(report,ensure_ascii=False,indent=2))+'</pre>')
    return dict(directory=str(output.relative_to(root)) if output.is_relative_to(root) else str(output),**report)


def project_saved_run(run_dir):
    root=Path(run_dir).resolve();config=json.loads((root/'config.json').read_text())
    candidates=json.loads((root/'candidates.json').read_text())
    result=export_gaussian_ply(root,config,candidates)
    report_path=root/'report.json'
    if report_path.exists():
        report=json.loads(report_path.read_text());report['gaussian_projection']=result;write_json(report_path,report)
        if report.get('mode')=='offline128':
            from .offline import html_report
            html_report(root,report)
    return result
