"""Geometry-only view scoring plus scalar clues from historical RGB.

This module cannot render or invoke models. Unvisited views are scored by
analytic projections and conservative geometry rays only. Coverage is a proxy
for useful surface observations, never a claim that small objects are resolved.
"""
import math
import numpy as np
from .camera import look_at


class ViewPolicy:
    def __init__(self, geometry, grid, config, camera_config):
        self.grid, self.config, self.camcfg = grid, config, camera_config
        self.rng = np.random.default_rng(config.get("seed", 7))
        # Uniform geometric samples, no appearance or semantic fields.
        keep = np.flatnonzero(grid.keep)
        count = min(config.get("coverage_samples", 160), len(keep))
        self.samples = geometry.means[self.rng.choice(keep, count, replace=False)] if count else np.empty((0, 3))
        self.geometric_seen = np.zeros(count, bool)
        self.effective_seen = np.zeros(count, bool)
        self.history = []
        self.verify_round = 0
        up=np.asarray(camera_config.get('world_up',[0,-1,0]),float)
        self.up_axis=int(np.argmax(np.abs(up)))
        self.horizontal_axes=[i for i in range(3) if i!=self.up_axis]
        self.free_samples=None
        self.free_seen=None

    def prepare_free_samples(self,camera):
        if self.free_samples is not None: return
        self.initial_height=float(camera.position[self.up_axis])
        points=self.grid.reachable_points(camera.position,self.config.get('free_coverage_samples',160),self.rng)
        self.free_samples=points
        self.free_seen=np.zeros(len(points),bool)

    def free_visibility(self,camera):
        points=self.free_samples
        p=(points-camera.position)@camera.c2w[:3,:3]
        z=p[:,2]
        uv=p[:,:2]/np.maximum(z[:,None],1e-6)*[camera.K[0,0],camera.K[1,1]]+[camera.K[0,2],camera.K[1,2]]
        mask=((z>.1)&(z<self.config.get('geometric_view_range',7))&
              (uv[:,0]>0)&(uv[:,0]<camera.width)&(uv[:,1]>0)&(uv[:,1]<camera.height))
        for i in np.flatnonzero(mask):
            mask[i]=self.grid.segment_free(camera.position,points[i])
        return mask

    def visibility(self, camera):
        p = (self.samples-camera.position) @ camera.c2w[:3, :3]
        z = p[:, 2]
        xy = p[:, :2] / np.maximum(z[:, None], 1e-6)
        uv = xy * [camera.K[0, 0], camera.K[1, 1]] + [camera.K[0, 2], camera.K[1, 2]]
        mask = ((z > .1) & (z < self.config.get("geometric_view_range", 7)) &
                (uv[:, 0] > 0) & (uv[:, 0] < camera.width) & (uv[:, 1] > 0) & (uv[:, 1] < camera.height))
        # Trim rays before obstacle surfaces; inflated geometry deliberately
        # underestimates visibility. This is not renderer-provided depth.
        for i in np.flatnonzero(mask):
            ray = self.samples[i]-camera.position
            distance = np.linalg.norm(ray)
            trim = self.grid.padding + 2*self.grid.voxel
            endpoint = camera.position + ray * max(0, 1-trim/max(distance, 1e-6))
            mask[i] = self.grid.segment_free(camera.position, endpoint)
        # Effective coverage additionally needs a resolvable configured object
        # scale and limited distance. Image quality is checked on arrived RGB.
        pixels = camera.K[0, 0] * self.config.get("coverage_object_size", .12) / np.maximum(z, 1e-6)
        effective = mask & (z < self.config.get("effective_view_range", 3)) & (
            pixels >= self.config.get("min_object_pixels", 12))
        return mask, effective

    def record_arrived(self, camera, rgb):
        import cv2
        self.prepare_free_samples(camera)
        geometric, effective = self.visibility(camera)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        contrast, sharpness = float(gray.std()), float(cv2.Laplacian(gray, cv2.CV_32F).var())
        quality = contrast >= self.config.get("min_rgb_contrast", 8) and sharpness >= self.config.get("min_rgb_sharpness", 2)
        if not quality:
            effective[:] = False
        self.geometric_seen |= geometric
        self.effective_seen |= effective
        self.history.append(camera)
        self.free_seen |= self.free_visibility(camera)
        return dict(geometric_surface_samples_seen=int(self.geometric_seen.sum()),
                    effective_surface_samples_seen=int(self.effective_seen.sum()),
                    total_surface_samples=len(self.samples), contrast=contrast,
                    sharpness=sharpness, rgb_quality_accepted=quality,
                    geometric_free_samples_seen=int(self.free_seen.sum()),total_free_samples=len(self.free_seen),
                    coverage_interpretation="surface proxy, not exhaustive small-object visibility")

    def explore(self, camera, memory, planner, remaining_distance=float("inf"),force_translate=False):
        self.prepare_free_samples(camera)
        count=self.config.get('position_samples',8)
        pool=self.grid.reachable_points(camera.position,max(160,count*24),self.rng)
        pool=pool[np.linalg.norm(pool-camera.position,axis=1)<=remaining_distance+1e-8]
        band=self.config.get('exploration_height_band',.24)
        layer=pool[np.abs(pool[:,self.up_axis]-self.initial_height)<=band]
        if len(layer): pool=layer
        minimum=self.config.get('min_explore_translation',1.5)
        if force_translate:
            pool=pool[np.linalg.norm(pool[:,self.horizontal_axes]-camera.position[self.horizontal_axes],axis=1)>=minimum]
        if len(pool):
            histories=np.array([h.position[self.horizontal_axes] for h in self.history] or [camera.position[self.horizontal_axes]])
            novelty=np.linalg.norm(pool[:,None,self.horizontal_axes]-histories[None],axis=2).min(1)
            # Mix spatially novel candidates with random reachable positions.
            distance=np.linalg.norm(pool-camera.position,axis=1)
            indices=np.argsort(np.minimum(novelty,3)-.10*distance)[::-1]
            chosen=[]
            for i in indices:
                if all(np.linalg.norm(pool[i,self.horizontal_axes]-p[self.horizontal_axes])>=.7 for p in chosen):
                    chosen.append(pool[i])
                if len(chosen)>=count: break
            positions=np.asarray(chosen).reshape(-1,3)
        else: positions=np.empty((0,3))
        # Current pose rotations remain useful but incur repetition penalty.
        if not force_translate: positions = np.vstack([positions, camera.position])
        up = np.array(self.camcfg.get("world_up", [0, -1, 0]), float)
        up /= np.linalg.norm(up)
        ref = np.array([0, 0, 1.])
        if abs(ref @ up) > .9:
            ref = np.array([1., 0, 0])
        e0 = ref-up*(ref@up)
        e0 /= np.linalg.norm(e0)
        e1 = np.cross(up, e0)
        scored = []
        for pos in positions:
            euclidean_cost = np.linalg.norm(pos-camera.position)
            if euclidean_cost > remaining_distance:
                continue
            directions=[]
            for theta in np.linspace(0,2*math.pi,self.config.get('yaw_samples',6),endpoint=False):
                horizontal=math.cos(theta)*e0+math.sin(theta)*e1
                for pitch in self.config.get('pitch_degrees',[0,-20]):
                    directions.append(math.cos(math.radians(pitch))*horizontal+math.sin(math.radians(pitch))*up)
            for f in directions:
                view = look_at(pos, f, up, camera.K, camera.width, camera.height)
                geometric, effective = self.visibility(view)
                novelty = int((geometric & ~self.geometric_seen).sum())
                useful = int((effective & ~self.effective_seen).sum())
                free_new=int((self.free_visibility(view)&~self.free_seen).sum())
                nearest=min((np.linalg.norm(h.position[self.horizontal_axes]-pos[self.horizontal_axes])
                             for h in self.history),default=0.)
                spatial=min(nearest/self.config.get('spatial_novelty_scale',2),1.)
                repeat = sum(math.exp(-np.linalg.norm(h.position-pos)) *
                             max(0, h.c2w[:3, 2] @ f)**4 for h in self.history)
                # Value derives only from candidate presence in previously arrived RGB.
                semantic = sum(min(1,o["candidate_count"])*math.exp(-np.linalg.norm(np.array(o["position"])-pos))
                               for o in memory.observations[-8:])
                score=(novelty+2*useful+30*free_new/max(1,len(self.free_seen))+15*spatial+
                       min(2,semantic)-.25*euclidean_cost-5*repeat)
                scored.append((score, view, dict(new_geometric_samples=novelty, new_effective_samples=useful,
                                                new_free_samples=free_new,spatial_novelty=spatial,
                                                history_rgb_value=semantic, repeat_penalty=repeat)))
        # Compute actual route costs only for top candidates, never render them.
        choices, failures = [], []
        planned_positions=[]
        for score, view, terms in sorted(scored,key=lambda x:x[0],reverse=True):
            if any(np.linalg.norm(view.position-p)<.5 for p in planned_positions): continue
            if len(planned_positions)>=self.config.get('plan_top_k',4): break
            planned_positions.append(view.position)
            try:
                path = planner.plan(camera.position, view.position)
                length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
                if length <= remaining_distance+1e-9:
                    details=planner.last_metadata
                    choices.append((score-.35*length,view,path,dict(terms,path_cost=length,
                                    planner_actual_backend=details.get('backend',planner.backend),
                                    planner_fallback_used=details.get('fallback_used',False),
                                    planner_primary_error=details.get('primary_error'))))
            except Exception as e:
                failures.append(dict(position=view.position.tolist(), error=repr(e)))
        if not choices:
            return None, None, dict(reason="no_feasible_view_within_path_budget", planning_failures=failures)
        score, view, path, terms = max(choices, key=lambda x:x[0])
        return view, path, dict(reason="geometry_novelty_rgb_history_route_cost", score=score, **terms,
                                force_translate=force_translate,planning_failures=failures)

    def verify(self, camera, candidate, planner, remaining_distance):
        anchor = candidate.sightings[0]
        baseline = self.config.get("verify_baseline", .5)
        right, down = anchor.camera.c2w[:3, 0], anchor.camera.c2w[:3, 1]
        forward = anchor.camera.c2w[:3, 2]
        box = np.array(anchor.detection.box)
        pixel = np.r_[.5*(box[:2]+box[2:]), 1.]
        direction = anchor.camera.c2w[:3, :3] @ np.linalg.inv(anchor.camera.K) @ pixel
        direction /= np.linalg.norm(direction)
        offsets = [right*baseline, -right*baseline, right*baseline*1.6,
                   -right*baseline*1.6, down*baseline, -down*baseline,
                   right*baseline-forward*baseline*.5, -right*baseline-forward*baseline*.5]
        if candidate.status=='pending_wider_view':
            offsets=[-direction*baseline*1.5,-direction*baseline*2,*offsets]
        offset_start = candidate.verify_attempts % len(offsets)
        failures = []
        for i in range(len(offsets)):
            pos = anchor.camera.position + offsets[(offset_start+i) % len(offsets)]
            if any(np.linalg.norm(pos-h.position) < baseline*.35 for h in self.history):
                continue
            try:
                path = planner.plan(camera.position, pos)
                length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
                if length > remaining_distance:
                    continue
                # A bearing only; no box-center depth or guessed target xyz.
                view = look_at(pos, direction, self.camcfg.get("world_up", [0, -1, 0]),
                               camera.K, camera.width, camera.height)
                return view, path, dict(reason="translate_for_rgb_correspondence_verification",
                                        anchor_observation=anchor.observation_id, candidate_id=candidate.id,
                                        requested_baseline=baseline,planning_failures=failures,
                                        planner_actual_backend=planner.last_metadata.get('backend',planner.backend),
                                        planner_fallback_used=planner.last_metadata.get('fallback_used',False),
                                        planner_primary_error=planner.last_metadata.get('primary_error'))
            except Exception as e:
                failures.append(dict(position=pos.tolist(), error=repr(e)))
        return None, None, dict(reason="no_feasible_translated_verification_pose", planning_failures=failures)
