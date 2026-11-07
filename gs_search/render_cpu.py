"""Slow, explicit CPU reference rasterizer, not gsplat CUDA validation.

Projects full anisotropic covariances, evaluates real SH, and performs ordered
front-to-back alpha compositing. Useful when GPU devices are not visible.
"""
import math
import numpy as np
from numba import njit
from scipy.spatial.transform import Rotation
from scipy.special import sph_harm_y


@njit(cache=True)
def composite(uv, conics, radius, opacity, colors, width, height):
    out = np.zeros((height, width, 3), np.float32)
    trans = np.ones((height, width), np.float32)
    for i in range(len(uv)):
        u, v = uv[i]
        r = radius[i]
        x1, x2 = max(0, int(math.floor(u-r))), min(width, int(math.ceil(u+r))+1)
        y1, y2 = max(0, int(math.floor(v-r))), min(height, int(math.ceil(v+r))+1)
        for y in range(y1, y2):
            for x in range(x1, x2):
                if trans[y,x] < 1e-4:
                    continue
                dx, dy = x+.5-u, y+.5-v
                power = -.5*(conics[i,0]*dx*dx+2*conics[i,1]*dx*dy+conics[i,2]*dy*dy)
                if power > 0:
                    continue
                alpha = min(.99, opacity[i]*math.exp(power))
                if alpha < 1/255:
                    continue
                weight = trans[y,x]*alpha
                for k in range(3):
                    out[y,x,k] += weight*colors[i,k]
                trans[y,x] *= 1-alpha
    return out


class CPUReferenceRenderer:
    backend = "cpu_reference_gaussian_alpha_compositing"

    def __init__(self, scene, config):
        self.scene, self.config, self.calls = scene, config, 0
        g = scene.geometry
        if len(g.means)*9*8 > config.get("max_cpu_covariance_bytes", 600_000_000):
            raise MemoryError("CPU covariance allocation exceeds configured limit")
        R = Rotation.from_quat(g.quats[:, [1,2,3,0]]).as_matrix()
        self.cov = (R*g.scales[:,None,:]**2) @ R.transpose(0,2,1)

    def render_batch(self, cameras):
        if len(cameras) > self.config.get("max_batch", 2):
            raise MemoryError("CPU render batch exceeds max_batch")
        return [self.render(c) for c in cameras]

    def render(self, camera):
        if camera.width*camera.height > self.config.get("max_pixels", 2_000_000):
            raise MemoryError("CPU render resolution exceeds max_pixels")
        g = self.scene.geometry
        means = (g.means-camera.position) @ camera.c2w[:3,:3]
        mask = (means[:,2] > self.config.get("near",.02)) & (means[:,2] < self.config.get("far",100)) & (g.opacity >= 1/255)
        ids = np.flatnonzero(mask)
        p = means[ids]
        R = camera.c2w[:3,:3].T
        cov = R[None] @ self.cov[ids] @ R.T[None]
        fx, fy = camera.K[0,0], camera.K[1,1]
        z = p[:,2]
        J = np.zeros((len(p),2,3))
        J[:,0,0], J[:,1,1] = fx/z, fy/z
        # Perspective Jacobian clamp follows standard 3DGS rasterizers.
        J[:,0,2] = -fx*np.clip(p[:,0]/z,-1.3*camera.width/(2*fx),1.3*camera.width/(2*fx))/z
        J[:,1,2] = -fy*np.clip(p[:,1]/z,-1.3*camera.height/(2*fy),1.3*camera.height/(2*fy))/z
        screen = J @ cov @ J.transpose(0,2,1) + np.eye(2)[None]*.3
        uv = p[:,:2]/z[:,None]*[fx,fy]+[camera.K[0,2],camera.K[1,2]]
        radius = np.ceil(3*np.sqrt(np.linalg.eigvalsh(screen)[:,-1]))
        visible = ((uv[:,0]+radius >= 0) & (uv[:,0]-radius < camera.width) &
                   (uv[:,1]+radius >= 0) & (uv[:,1]-radius < camera.height))
        ids, screen, uv, radius, z = [a[visible] for a in (ids,screen,uv,radius,z)]
        area = np.minimum(2*radius+1,camera.width)*np.minimum(2*radius+1,camera.height)
        if area.sum() > self.config.get("max_cpu_work_pixels", 300_000_000):
            raise MemoryError(f"CPU projected splat work {area.sum():.0f} exceeds max_cpu_work_pixels")
        directions = g.means[ids]-camera.position
        directions /= np.maximum(np.linalg.norm(directions,axis=1,keepdims=True),1e-12)
        theta, phi = np.arccos(directions[:,2].clip(-1,1)), np.arctan2(directions[:,1],directions[:,0])
        colors = np.zeros((len(ids),3))
        i = 0
        for l in range(self.scene.metadata['sh_degree']+1):
            for m in range(-l,l+1):
                harmonic = sph_harm_y(l,abs(m),theta,phi)
                real = harmonic.real if m==0 else np.sqrt(2)*(harmonic.imag if m<0 else harmonic.real)
                colors += real[:,None]*self.scene.sh[ids,i]
                i += 1
        colors = np.maximum(colors+.5,0).astype(np.float32)
        inverse = np.linalg.inv(screen)
        conics = np.stack([inverse[:,0,0],inverse[:,0,1],inverse[:,1,1]],axis=1)
        order = np.argsort(z)
        rgb = composite(uv[order],conics[order],radius[order],g.opacity[ids][order],colors[order],camera.width,camera.height)
        self.calls += 1
        return np.rint(rgb.clip(0,1)*255).astype(np.uint8)
