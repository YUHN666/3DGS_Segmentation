"""Geometry-only preparation of 16 safe camera stations and 128 RGB views.

No target text, appearance, renderer or semantic model enters site selection.
The supplied world-up is a coordinate convention, not inferred gravity.
"""
import copy
import hashlib
import html
import json
import traceback
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
from PIL import Image,ImageDraw
from .artifacts import write_json
from .grid_scan import expand_grid_cameras
from .navigation import OccupancyMap,PlanningError


DEFAULTS=dict(mode='auto',height_fractions=[.45,.35,.55,.65],
              max_candidates=4096,min_component_sites=8,min_spacing_voxels=2,
              scale_distances=True,reference_vertical_span=3.,max_ray_steps=512)

DISTANCES=[('navigation',['voxel_size','camera_radius','margin','move_step']),
           ('renderer',['near','far']),('localization',['near','far','min_depth_band']),
           ('gaussian_projection',['depth_abs']),('association',['min_baseline'])]


def inherit_captured_geometry(config,source_config):
    """Reuse calibrated distances/axes alongside captured images, not old targets."""
    config=copy.deepcopy(config)
    auto=config.setdefault('auto_views',{})
    reference=auto.get('reference_distances') or source_config.get('auto_views',{}).get('reference_distances')
    if reference is None:
        reference={section:{key:config[section][key] for key in keys if key in config.get(section,{})}
                   for section,keys in DISTANCES if section in config}
    auto['reference_distances']=reference
    if 'navigation' in source_config: config['navigation']=copy.deepcopy(source_config['navigation'])
    for section,keys in DISTANCES[1:]:
        for key in keys:
            if key in source_config.get(section,{}): config.setdefault(section,{})[key]=source_config[section][key]
    if 'camera' in source_config: config['camera']=copy.deepcopy(source_config['camera'])
    return config


def file_digest(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def basis_from_up(up):
    up=np.asarray(up,float)
    if up.shape!=(3,) or not np.isfinite(up).all() or np.linalg.norm(up)<1e-8:
        raise ValueError('world_up must be a finite nonzero 3-vector')
    up=up/np.linalg.norm(up)
    forward=np.array([0.,0.,1.])
    if abs(forward@up)>.95: forward=np.array([0.,1.,0.])
    forward-=up*(forward@up);forward/=np.linalg.norm(forward)
    return up,forward,np.cross(up,forward)


def _support_counts(points,grid,forward,side,max_steps):
    """Approximate horizontal enclosure; navigation boundaries are not walls."""
    directions=np.array([forward,-forward,side,-side]);n=len(points)
    hit=np.zeros((n,4),bool);active=np.ones((n,4),bool)
    clearance=grid.padding+grid.voxel
    steps=min(max_steps,int(np.ceil(np.linalg.norm(grid.upper-grid.lower)/grid.voxel)))
    for step in range(1,steps+1):
        probe=points[:,None,:]+directions[None,:,:]*(step*grid.voxel)
        index=np.floor((probe-grid.lower)/grid.voxel).astype(int)
        inside=((index>=0)&(index<grid.shape)).all(2)
        clipped=np.clip(index,0,grid.shape-1)
        occupied=grid.occupied[clipped[:,:,0],clipped[:,:,1],clipped[:,:,2]]
        boundary=((probe-grid.lower<clearance)|(grid.upper-probe<clearance)).any(2)
        hit|=active&inside&occupied&~boundary
        active&=inside&~occupied&~boundary
        if not active.any(): break
    return hit


def generate_layout(geometry,camera_config,navigation_config,options=None):
    """Return layout, adapted navigation, diagnostics and candidate sites."""
    opts={**DEFAULTS,**(options or {})}
    fractions=np.asarray(opts['height_fractions'],float)
    if fractions.ndim!=1 or not 1<=len(fractions)<=8 or not np.isfinite(fractions).all() or ((fractions<=0)|(fractions>=1)).any():
        raise ValueError('height_fractions must contain 1..8 finite values in (0,1)')
    if not 16<=opts['max_candidates']<=20000 or not 1<=opts['min_component_sites']<=1000:
        raise ValueError('Invalid automatic candidate/component budget')
    if not 1<=opts['min_spacing_voxels']<=10 or not 1<=opts['max_ray_steps']<=2048:
        raise ValueError('Invalid station spacing or ray budget')
    up,forward,side=basis_from_up(camera_config.get('world_up',[0,-1,0]))
    nav=copy.deepcopy(navigation_config)
    threshold=nav.get('opacity_threshold',.02)
    support=geometry.means[geometry.opacity>=threshold].astype(float)
    if len(support)<16: raise PlanningError('Insufficient retained Gaussian geometry for automatic cameras')
    vertical=support@up;floor,ceiling=np.percentile(vertical,[5,95]);span=float(ceiling-floor)
    if not np.isfinite(span) or span<=1e-8: raise PlanningError('Degenerate vertical extent; check world_up or scene scale')
    reference=float(opts['reference_vertical_span'])
    if not np.isfinite(reference) or reference<=0: raise ValueError('Invalid reference_vertical_span')
    unit=span/reference if opts['scale_distances'] else 1.
    for key in ['voxel_size','camera_radius','margin','move_step']:
        if key in nav: nav[key]=float(nav[key])*unit
    voxel=float(nav['voxel_size'])
    padding=max(.4*unit,2*voxel+nav['camera_radius']+nav['margin'])
    lower=geometry.means.min(0).astype(float)-padding
    upper=geometry.means.max(0).astype(float)+padding
    nav['bounds']=[lower.tolist(),upper.tolist()]
    # The occupancy constructor enforces max_cells; never drop obstacles or
    # silently make the grid coarser to satisfy a memory budget.
    grid=OccupancyMap(geometry,nav)
    coordinates=np.argwhere(~grid.occupied)
    if not len(coordinates): raise PlanningError('No collision-free space; inspect Gaussian scales and clearance parameters')
    points=grid.lower+(coordinates+.5)*voxel
    projected=np.column_stack([support@side,support@forward])
    inner=np.percentile(projected,[5,95],axis=0)
    # This bounds candidate camera sites, not the collision geometry.
    inner_padding=(inner[1]-inner[0])*.03;inner[0]+=inner_padding;inner[1]-=inner_padding
    horizontal=np.column_stack([points@side,points@forward])
    inner_mask=((horizontal>=inner[0])&(horizontal<=inner[1])).all(1)
    heights=points@up;trials=[];chosen=None
    minimum_sites=16
    for fraction in fractions:
        level=float(floor+span*fraction)
        keep=inner_mask&(np.abs(heights-level)<=voxel*.65)
        ids=np.flatnonzero(keep)
        if len(ids)>opts['max_candidates']:
            ids=ids[np.linspace(0,len(ids)-1,opts['max_candidates']).astype(int)]
        pool=points[ids]
        hits=_support_counts(pool,grid,forward,side,opts['max_ray_steps']) if len(pool) else np.empty((0,4),bool)
        # Prefer at least three wall/obstacle directions; relax only to an
        # opposite pair, recording the selected support rule explicitly.
        masks=[(hits.sum(1)>=3,'at_least_three_horizontal_directions'),
               ((hits[:,0]&hits[:,1])|(hits[:,2]&hits[:,3]),'opposed_horizontal_directions')]
        trial=dict(fraction=float(fraction),height_along_up=level,candidates_before_support=len(pool),support_counts=[])
        for good,rule in masks:
            candidate_ids=ids[good]
            labels=grid.labels[tuple(coordinates[candidate_ids].T)]
            counts=np.bincount(labels,minlength=grid.components+1)
            good_components=np.flatnonzero(counts>=opts['min_component_sites']);good_components=good_components[good_components>0]
            candidate_ids=candidate_ids[np.isin(labels,good_components)]
            trial['support_counts'].append(dict(rule=rule,candidates=len(candidate_ids)))
            if len(candidate_ids)>=minimum_sites:
                chosen=(candidate_ids,rule,level,float(fraction));break
        trials.append(trial)
        if chosen is not None: break
    if chosen is None:
        raise PlanningError('Cannot find 16 supported safe camera sites. Check world_up, clearance, height_fractions or incomplete/open geometry; no fallback to unsafe sites')
    ids,rule,level,fraction=chosen;pool=points[ids];indices=coordinates[ids]
    labels=grid.labels[tuple(indices.T)]
    clearance=distance_transform_edt(~grid.occupied)*voxel
    quality=clearance[tuple(indices.T)]
    component_counts={int(label):int((labels==label).sum()) for label in np.unique(labels)}
    components=sorted(component_counts,key=lambda label:(-component_counts[label],label))
    selected=[]
    # Offline views are independent camera placements, so separate free-space
    # components may be sampled; this never claims an executable travel path.
    for label in components[:16]:
        group=np.flatnonzero(labels==label)
        for previous in selected:
            delta=pool[group]-pool[previous]
            distance=np.sqrt((delta@side)**2+(delta@forward)**2+.25*(delta@up)**2)
            group=group[distance>=voxel*opts['min_spacing_voxels']]
        if not len(group): continue
        center=pool[group].mean(0)
        rank=quality[group]-.01*np.linalg.norm(pool[group]-center,axis=1)
        selected.append(int(group[np.argmax(rank)]))
    nearest=np.full(len(pool),np.inf)
    def update(index):
        nonlocal nearest
        delta=pool-pool[index]
        distance=np.sqrt((delta@side)**2+(delta@forward)**2+.25*(delta@up)**2)
        nearest=np.minimum(nearest,distance)
    for index in selected: update(index)
    while len(selected)<16:
        score=nearest*(.8+.2*quality/max(float(quality.max()),1e-9))
        score[nearest<voxel*opts['min_spacing_voxels']]=-1
        score[selected]=-1
        index=int(np.argmax(score))
        if score[index]<0: raise PlanningError('Insufficient distinct safe stations at configured spacing')
        selected.append(index);update(index)
    stations=pool[selected]
    for point in stations: grid.require_free(point,'automatic camera')
    width=int(camera_config.get('width',800));height=int(camera_config.get('height',600))
    fov=float(camera_config.get('fov_degrees',75))
    if not 8<=width<=8192 or not 8<=height<=8192 or not np.isfinite(fov) or not 5<=fov<=150:
        raise ValueError('Invalid automatic camera resolution/FOV')
    focal=.5*width/np.tan(np.radians(fov)/2)
    layout=dict(width=width,height=height,K=[[focal,0,width/2],[0,focal,height/2],[0,0,1]],
        up=up.tolist(),forward=forward.tolist(),yaw_degrees=list(range(0,360,45)),
        camera_positions=[dict(name=f'p{i:02d}',position=p.tolist()) for i,p in enumerate(stations)])
    expand_grid_cameras(layout)
    diagnostics=dict(method='geometry_free_space_enclosure_and_farthest_sites',options=opts,
        world_up=up.tolist(),world_up_source='configured convention; gravity cannot be established from an arbitrary PLY',
        horizontal_basis=[side.tolist(),forward.tolist()],vertical_percentiles_05_95=[float(floor),float(ceiling)],
        vertical_span=span,distance_parameter_scale=unit,
        scale_meaning='Relative to robust vertical span; not a physical metre calibration',
        selected_fraction=fraction,height_along_up=level,height_trials=trials,support_rule=rule,
        geometry=grid.metadata,scene_bounds_source='All valid Gaussian centers plus clearance padding; no size filtering',
        candidate_bounds_in_horizontal_basis=inner.tolist(),candidate_pool=len(pool),
        component_candidates=component_counts,selected_components=labels[selected].tolist(),
        omitted_components=[label for label in components if label not in labels[selected]],
        min_station_spacing=voxel*opts['min_spacing_voxels'],all_positions_collision_free=True,
        independent_offline_placements=True,navigation_path_claimed=False,
        semantic_inputs_used=False,preview_RGB_used_for_selection=False,
        limitation='Geometric support and 16 positions are coverage proxies, not exhaustive visual/object coverage')
    return layout,nav,diagnostics,pool


def validate_binding(layout,scene_config):
    binding=layout.get('preparation',{}).get('binding')
    if binding:
        path=Path(scene_config['path']).resolve()
        if str(path)!=binding['path'] or scene_config.get('world_scale',1)!=binding['world_scale'] or file_digest(path)!=binding['sha256']:
            raise ValueError('Camera layout belongs to different/modified geometry; use --camera-mode auto')


def prepare_scene(scene,config,directory):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=False)
    config=copy.deepcopy(config)
    auto=config.setdefault('auto_views',{})
    reference=auto.get('reference_distances')
    if reference and auto.get('scale_distances',True):
        for section,values in reference.items(): config.setdefault(section,{}).update(values)
    else:
        reference={section:{key:config[section][key] for key in keys if key in config.get(section,{})}
                   for section,keys in DISTANCES if section in config}
    auto['reference_distances']=reference
    try:
        layout,nav,meta,pool=generate_layout(scene.geometry,config['camera'],config['navigation'],auto)
    except Exception as exc:
        write_json(directory/'preparation_error.json',dict(scene=scene.metadata,options=auto,
            error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())))
        raise
    path=Path(config['scene']['path']).resolve()
    meta['binding']=dict(path=str(path),sha256=file_digest(path),world_scale=config['scene'].get('world_scale',1))
    layout['preparation']=meta
    write_json(directory/'layout.json',layout);write_json(directory/'preparation.json',meta)
    config['navigation']=nav
    config.setdefault('auto_views',{})['mode']='fixed'
    config['camera'].pop('poses_file',None)
    config['camera'].update(initial_pose_index=None,start=layout['camera_positions'][0]['position'],
                            world_up=layout['up'],direction=layout['forward'])
    config.setdefault('policy',{}).setdefault('grid128',{}).update(positions_file=str(directory/'layout.json'),position_order='reference')
    config['policy']['strategy']='grid128'
    scale=meta['distance_parameter_scale']
    if config.get('auto_views',{}).get('scale_distances',True):
        for section,keys in [('renderer',['near','far']),('localization',['near','far','min_depth_band']),
                             ('gaussian_projection',['depth_abs']),('association',['min_baseline'])]:
            for key in keys:
                if key in config.get(section,{}): config[section][key]*=scale
    diagonal=float(np.linalg.norm(np.subtract(*nav['bounds'])))
    config['renderer']['far']=max(config['renderer'].get('far',60),1.25*diagonal)
    if 'localization' in config: config['localization']['far']=config['renderer']['far']
    if config['camera']['width']*config['camera']['height']>config['renderer'].get('max_pixels',1500000):
        raise MemoryError('Automatic camera resolution exceeds renderer.max_pixels')
    write_json(directory/'cameras.json',dict(cameras=[c.as_dict() for group in expand_grid_cameras(layout) for c in group]))
    side,forward=np.asarray(meta['horizontal_basis']);projected=np.column_stack([pool@side,pool@forward])
    low,high=projected.min(0),projected.max(0);span=np.maximum(high-low,1e-9)
    canvas=Image.new('RGB',(840,640),(20,26,37));draw=ImageDraw.Draw(canvas)
    def pixels(points): return (30+(points-low)/span*[780,560]).astype(int)
    for x,y in pixels(projected): draw.ellipse((x-1,610-y,x+1,612-y),fill='#486b84')
    positions=np.asarray([s['position'] for s in layout['camera_positions']])
    for i,(x,y) in enumerate(pixels(np.column_stack([positions@side,positions@forward]))):
        y=611-y;draw.ellipse((x-5,y-5,x+5,y+5),fill='#ff5858');draw.text((x+6,y-6),str(i),fill='white')
    draw.text((18,615),'Blue: eligible geometry-only sites | Red: 16 chosen stations | No motion path',fill='white')
    canvas.save(directory/'stations.png')
    (directory/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>自动相机准备</h1>'
        '<p>只使用几何选点；16个独立位置、每处8方向。未运行目标识别。</p><img width="840" src="stations.png">'
        '<p><a href="layout.json">视角布局</a> · <a href="cameras.json">128个相机参数</a> · '
        '<a href="preparation.json">几何检查与场景绑定</a></p><pre>'+html.escape(json.dumps(meta,ensure_ascii=False,indent=2))+'</pre>')
    return config,meta
