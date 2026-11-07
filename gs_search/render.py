"""RGB-only renderer for active search; opt-in offline RGB-D is separate."""
import numpy as np


def make_renderer(scene, config):
    backend = config.get("backend", "gsplat")
    if backend == "gsplat":
        return GSplatRenderer(scene, config)
    if backend == "cpu_reference":
        from .render_cpu import CPUReferenceRenderer
        return CPUReferenceRenderer(scene, config)
    raise ValueError(f"Unknown renderer backend: {backend}")


class GSplatRenderer:
    backend = "gsplat"

    def __init__(self, scene, config):
        import torch
        self.torch = torch
        self.device = config.get("device", "cuda:0")
        if not torch.cuda.is_available():
            raise RuntimeError("gsplat requires visible CUDA GPU. Run outside sandbox with CUDA_VISIBLE_DEVICES=2,5")
        # Build/load kernels during initialization, before the search clock
        # starts. This performs no rendering of candidate viewpoints.
        from gsplat.cuda._backend import _C
        if _C is None:
            raise RuntimeError("gsplat CUDA extension unavailable")
        self.config = config
        g = scene.geometry
        self.means, self.scales, self.quats, self.opacity, self.sh = [
            torch.as_tensor(a, dtype=torch.float32, device=self.device).contiguous()
            for a in (g.means, g.scales, g.quats, g.opacity, scene.sh)]
        self.degree = scene.metadata["sh_degree"]
        self.calls = 0
        self.max_batch = config.get("max_batch", 2)

    def render(self, camera):
        return self.render_batch([camera])[0]

    def render_batch(self, cameras):
        if not cameras:
            return []
        if len(cameras) > self.max_batch:
            raise MemoryError(f"Batch exceeds max_batch={self.max_batch}")
        w, h = cameras[0].width, cameras[0].height
        if w * h * len(cameras) > self.config.get("max_pixels", 2_000_000):
            raise MemoryError("Render exceeds max_pixels")
        if any((c.width, c.height) != (w, h) for c in cameras):
            raise ValueError("Batch cameras must share resolution")
        from gsplat import rasterization
        t = self.torch
        with t.inference_mode():
            rgb, _, _ = rasterization(
                means=self.means, quats=self.quats, scales=self.scales,
                opacities=self.opacity, colors=self.sh,
                viewmats=t.as_tensor(np.stack([np.linalg.inv(c.c2w) for c in cameras]), device=self.device, dtype=t.float32),
                Ks=t.as_tensor(np.stack([c.K for c in cameras]), device=self.device, dtype=t.float32),
                width=w, height=h, sh_degree=self.degree, packed=True,
                near_plane=self.config.get("near", .02), far_plane=self.config.get("far", 100),
                render_mode="RGB", backgrounds=t.zeros((len(cameras), 3), device=self.device))
            t.cuda.synchronize(self.device)
        self.calls += len(cameras)
        return list((rgb.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))
