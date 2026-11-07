"""Post-VLM geometry localization. No semantic classifier or RGB thresholds.

Single-view output is a visible-surface estimate, not an object's complete
center/volume. Different observations remain separate; same class never merges.
"""
from pathlib import Path
import numpy as np
from plyfile import PlyData,PlyElement


class SAM2BoxSegmenter:
    backend='local_sam2_box_prompt'
    def __init__(self,config):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.torch=torch;self.config=config;self.calls=0
        checkpoint=Path(config['checkpoint'])
        if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
        self.predictor=SAM2ImagePredictor(build_sam2(config['model_config'],str(checkpoint),device=config.get('device','cuda:0')))

    def segment(self,rgb,box):
        with self.torch.inference_mode():
            self.predictor.set_image(np.ascontiguousarray(rgb).copy())
            masks,scores,_=self.predictor.predict(box=np.asarray(box,np.float32),multimask_output=False)
        self.calls+=1
        index=int(np.argmax(scores))
        return masks[index].astype(bool),dict(backend=self.backend,mask_model_score=float(scores[index]),
                                             score_is_calibrated_probability=False)


def backproject_mask(mask,expected_z,alpha,camera,config):
    """Use actual mask pixels plus expected Z; never the detection-box center."""
    mask=np.asarray(mask,bool);depth=np.asarray(expected_z);opacity=np.asarray(alpha)
    if mask.shape!=(camera.height,camera.width) or depth.shape!=mask.shape or opacity.shape!=mask.shape:
        raise ValueError('Mask/depth/alpha dimensions must match camera')
    minimum=config.get('min_points',30)
    if not 1<=minimum<=1_000_000: raise ValueError('Invalid min_points')
    if mask.sum()>mask.size*config.get('max_mask_fraction',.5):
        return dict(status='UNRESOLVED',reason='mask_covers_too_much_of_frame',target_xyz=None),None,None
    valid=mask&np.isfinite(depth)&np.isfinite(opacity)&(depth>config.get('near',.02))&(
        depth<config.get('far',60))&(opacity>=config.get('min_alpha',.6))
    count=int(valid.sum())
    if count<minimum: return dict(status='UNRESOLVED',reason='insufficient_valid_mask_depth',valid_pixels=count,target_xyz=None),None,None
    yy,xx=np.nonzero(valid);z=depth[yy,xx].astype(float)
    # Depth median/MAD removes tails from transparency, foreground/background
    # mixtures and isolated splats. It does not establish object identity.
    median=float(np.median(z));mad=float(np.median(np.abs(z-median)))
    tolerance=max(config.get('min_depth_band',.05),config.get('relative_depth_band',.15)*median,3*1.4826*mad)
    keep=np.abs(z-median)<=tolerance;xx,yy,z=xx[keep],yy[keep],z[keep]
    if len(z)<minimum: return dict(status='UNRESOLVED',reason='depth_filter_removed_support',target_xyz=None),None,None
    # gsplat pixel centers are (u+.5,v+.5), K is OpenCV pinhole.
    pixels=np.column_stack([xx+.5,yy+.5,np.ones(len(xx))])
    camera_points=(pixels@np.linalg.inv(camera.K).T)*z[:,None]
    points=camera_points@camera.c2w[:3,:3].T+camera.position
    maximum=config.get('max_points',20000)
    if not 1<=maximum<=1_000_000: raise ValueError('Invalid max_points')
    if len(points)>maximum:
        take=np.linspace(0,len(points)-1,maximum).astype(int);points=points[take];xx,yy=xx[take],yy[take]
    position=np.median(points,axis=0)
    result=dict(status='LOCALIZED',method='sam2_mask_rendered_expected_z',target_xyz=position.tolist(),
                position_kind='median_of_visible_surface_points',coordinate_units='scene units',
                points=len(points),valid_pixels_before_filter=count,
                visible_surface_bounds_05_95=np.percentile(points,[5,95],axis=0).tolist(),
                depth_median=median,depth_mad=mad,pixel_center_offset=.5,
                limitation='Approximate visible surface; not a calibrated object center or complete 3D box')
    return result,points,np.column_stack([yy,xx])


def write_colored_cloud(path,points,colors):
    points=np.asarray(points);colors=np.asarray(colors)
    if points.ndim!=2 or points.shape[1]!=3 or colors.shape!=points.shape or not np.isfinite(points).all():
        raise ValueError('Invalid colored point cloud')
    data=np.empty(len(points),dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])
    for i,name in enumerate(['x','y','z']): data[name]=points[:,i]
    for i,name in enumerate(['red','green','blue']): data[name]=colors[:,i]
    PlyData([PlyElement.describe(data,'vertex')],text=False).write(str(path))


def export_scene_html(root,config,candidates):
    """Optional offline interactive plot of bounded scene centers and target clouds.

    This is a point-cloud overview, not a new 3DGS renderer or a perception input.
    """
    from .scene import load_scene
    import plotly.graph_objects as go
    root=Path(root);scene=load_scene(config['scene']);g=scene.geometry
    lower,upper=np.asarray(config['navigation']['bounds'],float)
    keep=(g.opacity>=config['navigation'].get('opacity_threshold',.02))&(
        (g.means>=lower)&(g.means<=upper)).all(1)
    indices=np.flatnonzero(keep);maximum=config.get('offline',{}).get('max_scene_plot_points',30000)
    if not 1<=maximum<=100000: raise ValueError('max_scene_plot_points must be within 1..100000')
    if len(indices)>maximum: indices=indices[np.linspace(0,len(indices)-1,maximum).astype(int)]
    xyz=g.means[indices].round(4)
    colors=((.5+.28209479177387814*scene.sh[indices,0]).clip(0,1)*255).astype(np.uint8)
    fig=go.Figure(go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='markers',name='Scene Gaussian centers',
        marker=dict(size=1,opacity=.22,color=[f'rgb({r},{gg},{b})' for r,gg,b in colors]),hoverinfo='skip'))
    for c in candidates:
        if c.get('localization',{}).get('status')!='LOCALIZED': continue
        vertex=PlyData.read(str(root/c['localization']['pointcloud']))['vertex'].data
        points=np.column_stack([vertex[a] for a in ['x','y','z']])
        fig.add_trace(go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2],mode='markers',name=c['id'],
                                  marker=dict(size=3,opacity=.9,color='#ef4444')))
        pos=c['target_xyz']
        fig.add_trace(go.Scatter3d(x=[pos[0]],y=[pos[1]],z=[pos[2]],mode='markers',
            hovertext=[c['id']+' visible-surface median'],hoverinfo='text',showlegend=False,
            marker=dict(size=7,color='#f59e0b',symbol='diamond')))
    up=config.get('camera',{}).get('world_up',[0,-1,0])
    fig.update_layout(title='Offline localized targets · scene units · Gaussian-center overview',
        scene=dict(aspectmode='data',xaxis_title='X',yaxis_title='Y',zaxis_title='Z',
                   camera=dict(up=dict(zip(['x','y','z'],up)),eye=dict(x=1.2,y=-1.4,z=1.2))),
        margin=dict(l=0,r=0,t=50,b=0))
    path=root/'localization_3d.html';fig.write_html(str(path),include_plotlyjs=True,full_html=True)
    return dict(html=path.name,scene_points=len(indices),
                display_only_filter='opacity and configured bounds; does not affect localization or recognition',
                representation='Gaussian centers plus mask/depth target surface points, not Gaussian splat rendering')
