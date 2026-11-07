"""Offline-only RGB/geometry acquisition; active RGB renderer stays unchanged.

Expected camera-Z and alpha are saved for explicitly opted-in localization.
Recognition receives only the uint8 RGB returned in a separate field.
"""
import numpy as np
from .render import GSplatRenderer


class OfflineRGBDRenderer(GSplatRenderer):
    backend='gsplat_offline_rgb_expected_z'

    def render_batch_rgbd(self,cameras):
        if not cameras: return []
        if len(cameras)>self.max_batch: raise MemoryError('RGBD batch exceeds max_batch')
        w,h=cameras[0].width,cameras[0].height
        if w*h*len(cameras)>self.config.get('max_pixels',1_500_000): raise MemoryError('RGBD pixels exceed max_pixels')
        if any((c.width,c.height)!=(w,h) for c in cameras): raise ValueError('Mixed RGBD resolution')
        from gsplat import rasterization
        t=self.torch
        with t.inference_mode():
            rendered,alpha,_=rasterization(means=self.means,quats=self.quats,scales=self.scales,
                opacities=self.opacity,colors=self.sh,
                viewmats=t.as_tensor(np.stack([np.linalg.inv(c.c2w) for c in cameras]),device=self.device,dtype=t.float32),
                Ks=t.as_tensor(np.stack([c.K for c in cameras]),device=self.device,dtype=t.float32),
                width=w,height=h,sh_degree=self.degree,packed=True,
                near_plane=self.config.get('near',.02),far_plane=self.config.get('far',100),render_mode='RGB+ED',
                backgrounds=t.zeros((len(cameras),3),device=self.device))
            t.cuda.synchronize(self.device)
        rgb=(rendered[...,:3].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)
        depth=rendered[...,3].cpu().numpy();opacity=alpha[...,0].cpu().numpy()
        self.calls+=len(cameras)
        return [dict(rgb=rgb[i],expected_z=depth[i],alpha=opacity[i]) for i in range(len(cameras))]
