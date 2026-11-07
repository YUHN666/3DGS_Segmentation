"""Persistent geometry-only stations and surface coverage. No renderer/models.

Visibility uses an uninflated Gaussian occupancy proxy; motion always uses the
original inflated collision map. Coverage means sampled view quality, never
object absence or recognition. Only record_arrived can mark evidence observed.
"""
import math
import copy
import numpy as np
from numba import njit
from .camera import look_at
from .navigation import OccupancyMap, PlanningError
from .policy import ViewPolicy


@njit(cache=True)
def clear_rays(origin, endpoints, occupied, lower, voxel):
    result=np.ones(len(endpoints),np.bool_)
    for i in range(len(endpoints)):
        d=endpoints[i]-origin
        steps=max(1,int(math.ceil(np.sqrt((d*d).sum())/(voxel*.4))))
        for j in range(steps+1):
            p=origin+d*(j/steps)
            cell=np.floor((p-lower)/voxel).astype(np.int64)
            if (cell<0).any() or cell[0]>=occupied.shape[0] or cell[1]>=occupied.shape[1] or cell[2]>=occupied.shape[2]:
                result[i]=False;break
            if occupied[cell[0],cell[1],cell[2]]:
                result[i]=False;break
    return result


class CoveragePolicy(ViewPolicy):
    """A finite, deterministic station set with a coarse pass and gap filling."""
    def __init__(self,geometry,grid,config,camera_config):
        super().__init__(geometry,grid,config,camera_config)
        self.options=config.get('coverage',{})
        o=self.options
        for name,default in [('station_spacing',1.8),('patch_size',.4),('effective_range',3.),('view_range',7.)]:
            if not np.isfinite(o.get(name,default)) or o.get(name,default)<=0:
                raise ValueError(f'coverage.{name} must be positive')
        for name,default,limit in [('max_stations',128,512),('max_patches',3072,16384),('yaw_samples',8,36),('views_per_visit',3,100)]:
            value=o.get(name,default)
            if not isinstance(value,int) or not 1<=value<=limit: raise ValueError(f'Invalid coverage.{name}')
        self.pitches=o.get('pitch_degrees',[0,-40,30])
        if not self.pitches or not np.isfinite(self.pitches).all() or max(abs(float(p)) for p in self.pitches)>=85:
            raise ValueError('Coverage pitch_degrees must be finite and strictly between -85 and 85')
        if not 0<o.get('effective_fraction_goal',.9)<=1: raise ValueError('Invalid effective_fraction_goal')
        raw_config=dict(grid.config,camera_radius=0.,margin=0.)
        self.visibility_grid=OccupancyMap(geometry,raw_config)
        self._surface_patches()
        self.stations=[];self.views=[];self.masks=[];self.current_station=None
        self.visit_start=0;self.initialized=False;self.last_coverage=None
        self.first_seen=np.full(len(self.patches),-1,int)
        self.first_effective=np.full(len(self.patches),-1,int)
        self.potential=np.zeros(len(self.patches),bool)
        self.station_failures=[]

    def _surface_patches(self):
        g=self.visibility_grid;o=self.options
        points=[];normals=[]
        for axis in range(3):
            for sign in [-1,1]:
                neighbor=np.roll(g.occupied,-sign,axis=axis)
                exposed=g.occupied&~neighbor
                # Navigation volume boundaries are not reconstructed surfaces.
                exposed[0,:,:]=exposed[-1,:,:]=False
                exposed[:,0,:]=exposed[:,-1,:]=False
                exposed[:,:,0]=exposed[:,:,-1]=False
                cells=np.argwhere(exposed)
                n=np.zeros(3);n[axis]=sign
                points.append(g.lower+(cells+.5)*g.voxel+n*g.voxel*.5)
                normals.append(np.tile(n,(len(cells),1)))
        p=np.vstack(points);n=np.vstack(normals)
        if not len(p):
            self.patches=p;self.normals=n;self.weights=np.empty(0)
            self.patch_metadata=dict(exposed_faces=0,patches=0,sampling='uniform spatial cells and face orientation')
            return
        size=o.get('patch_size',.4)
        keys=np.c_[np.floor((p-g.lower)/size).astype(int),n.astype(int)]
        _,first,inverse,counts=np.unique(keys,axis=0,return_index=True,return_inverse=True,return_counts=True)
        # Mean of a patch can fall inside geometry; retain an actual exposed face.
        p,n=p[first],n[first];weights=counts*g.voxel**2
        before=len(p);cap=o.get('max_patches',3072)
        if before>cap:
            take=np.linspace(0,before-1,cap).astype(int)
            p,n,weights=p[take],n[take],weights[take]
        self.patches,self.normals,self.weights=p,n,weights
        self.patch_metadata=dict(exposed_faces=int(counts.sum()),spatial_patches_before_cap=before,
                                patches=len(p),max_patches=cap,patch_size=size,
                                cap_applied=before>cap,sampling='uniform spatial cells and face orientation',
                                visibility='uninflated Gaussian sigma-AABB voxel proxy; finite ray samples',
                                coverage_is_sampled_proxy=True)

    def initialize(self,camera):
        if self.initialized: return
        g=self.grid;o=self.options
        label=g.labels[g.require_free(camera.position)]
        cells=np.argwhere(g.labels==label);points=g.lower+(cells+.5)*g.voxel
        up=np.asarray(self.camcfg.get('world_up',[0,-1,0]),float);up/=np.linalg.norm(up)
        offsets=o.get('height_offsets',[0])
        if not offsets or len(offsets)>16 or not np.isfinite(offsets).all(): raise ValueError('Invalid coverage.height_offsets')
        spacing=o.get('station_spacing',1.8)
        stations=[camera.position]
        for height in offsets:
            target=camera.position+up*height
            subset=points[np.abs(points[:,self.up_axis]-target[self.up_axis])<=o.get('height_band',.24)]
            if not len(subset): continue
            keys=np.floor((subset[:,self.horizontal_axes]-g.lower[self.horizontal_axes])/spacing).astype(int)
            unique,inverse=np.unique(keys,axis=0,return_inverse=True)
            for i,key in enumerate(unique):
                choices=subset[inverse==i]
                center=g.lower[self.horizontal_axes]+(key+.5)*spacing
                score=((choices[:,self.horizontal_axes]-center)**2).sum(1)+4*(choices[:,self.up_axis]-target[self.up_axis])**2
                point=choices[np.argmin(score)]
                height_separation=max(g.voxel*1.5,.2)
                if all(np.linalg.norm(point-p)>=spacing*.45 or
                       abs(point[self.up_axis]-p[self.up_axis])>=height_separation for p in stations):
                    stations.append(point)
        if len(stations)>o.get('max_stations',128):
            raise MemoryError(f'{len(stations)} coverage stations exceeds max_stations; increase station_spacing or cap explicitly')
        cache_bytes=len(stations)*o.get('yaw_samples',8)*len(self.pitches)*len(self.patches)*2
        if cache_bytes>o.get('max_visibility_cache_mb',128)*1024**2:
            raise MemoryError('Coverage visibility cache exceeds max_visibility_cache_mb; reduce stations/views/patches')
        ray_steps=len(stations)*len(self.patches)*float(np.linalg.norm(g.shape))/.4
        if ray_steps>o.get('max_visibility_ray_steps',250_000_000):
            raise MemoryError('Coverage ray work exceeds max_visibility_ray_steps; increase station_spacing/patch_size')
        ref=np.array([0.,0.,1.])
        if abs(ref@up)>.9: ref=np.array([1.,0.,0.])
        e0=ref-up*(ref@up);e0/=np.linalg.norm(e0);e1=np.cross(up,e0)
        for i,point in enumerate(stations):
            views=[];masks=[]
            delta=point-self.patches
            distance=np.linalg.norm(delta,axis=1)
            facing=np.sum(delta*self.normals,axis=1)>self.visibility_grid.voxel*.02
            possible=facing&(distance<o.get('view_range',7.))
            raw=np.zeros(len(self.patches),bool)
            indices=np.flatnonzero(possible)
            raw[indices]=clear_rays(point,self.patches[indices]+self.normals[indices]*self.visibility_grid.voxel*.01,
                                     self.visibility_grid.occupied,self.visibility_grid.lower,self.visibility_grid.voxel)
            for theta in np.linspace(0,2*np.pi,o.get('yaw_samples',8),endpoint=False):
                horizontal=math.cos(theta)*e0+math.sin(theta)*e1
                for pitch in self.pitches:
                    angle=math.radians(pitch)
                    view=look_at(point,math.cos(angle)*horizontal+math.sin(angle)*up,up,camera.K,camera.width,camera.height)
                    masks.append(self._project(view,raw));views.append(view)
            self.stations.append(dict(id=i,position=point.tolist(),view_first_observed=[-1]*len(views),planning_failures=[]))
            self.views.append(views);self.masks.append(masks)
            for _,effective in masks: self.potential|=effective
        self.initialized=True
        self.current_station=0

    def _project(self,camera,visible):
        p=(self.patches-camera.position)@camera.c2w[:3,:3];z=p[:,2]
        uv=p[:,:2]/np.maximum(z[:,None],1e-8)*[camera.K[0,0],camera.K[1,1]]+[camera.K[0,2],camera.K[1,2]]
        seen=visible&(z>.05)&(uv[:,0]>=2)&(uv[:,0]<camera.width-2)&(uv[:,1]>=2)&(uv[:,1]<camera.height-2)
        pixels=camera.K[0,0]*self.config.get('coverage_object_size',.12)/np.maximum(z,1e-8)
        effective=seen&(np.linalg.norm(p,axis=1)<=self.options.get('effective_range',3.))&(pixels>=self.config.get('min_object_pixels',12))
        return seen,effective

    def _visible_at(self,camera):
        delta=camera.position-self.patches
        possible=(np.sum(delta*self.normals,axis=1)>self.visibility_grid.voxel*.02)&(np.linalg.norm(delta,axis=1)<self.options.get('view_range',7.))
        indices=np.flatnonzero(possible);raw=np.zeros(len(self.patches),bool)
        raw[indices]=clear_rays(camera.position,self.patches[indices]+self.normals[indices]*self.visibility_grid.voxel*.01,
                                self.visibility_grid.occupied,self.visibility_grid.lower,self.visibility_grid.voxel)
        return self._project(camera,raw)

    def record_arrived(self,camera,rgb):
        self.initialize(camera)
        # Retain legacy diagnostics independently, for old report readers.
        legacy=super().record_arrived(camera,rgb)
        obs=len(self.history)-1
        seen,effective=self._visible_at(camera)
        if not legacy['rgb_quality_accepted']: effective[:]=False
        new_seen=np.flatnonzero((self.first_seen<0)&seen)
        new_effective=np.flatnonzero((self.first_effective<0)&effective)
        self.first_seen[new_seen]=obs
        self.first_effective[new_effective]=obs
        station_updates=[]
        for i,station in enumerate(self.stations):
            if np.linalg.norm(np.array(station['position'])-camera.position)>1e-4: continue
            for j,view in enumerate(self.views[i]):
                if np.allclose(camera.c2w[:3,:3],view.c2w[:3,:3],atol=1e-5) and np.allclose(camera.K,view.K):
                    if station['view_first_observed'][j]<0:
                        station['view_first_observed'][j]=obs
                        station_updates.append([i,j])
        self.last_coverage=dict(legacy,**self.summary(),coverage_update=dict(
            seen=new_seen.tolist(),effective=new_effective.tolist(),station_views=station_updates))
        return self.last_coverage

    def summary(self):
        goal=self.options.get('effective_fraction_goal',.9)
        denom=float(self.weights[self.potential].sum())
        covered=float(self.weights[self.potential&(self.first_effective>=0)].sum())
        fraction=covered/denom if denom else 0.
        quota=self.options.get('views_per_visit',3)
        visited=[sum(v>=0 for v in s['view_first_observed']) for s in self.stations]
        return dict(strategy='coverage',stations_total=len(self.stations),stations_visited=sum(n>0 for n in visited),
                    stations_coarse_complete=sum(n>=quota for n in visited),station_views_observed=sum(visited),
                    surface_patches=len(self.patches),surface_patches_seen=int((self.first_seen>=0).sum()),
                    surface_patches_effective=int((self.first_effective>=0).sum()),
                    sampled_potentially_effective_patches=int(self.potential.sum()),
                    patches_without_sampled_effective_view=int((~self.potential).sum()),
                    effective_fraction_of_sampled_observable=fraction,effective_fraction_goal=goal,
                    coarse_pass_complete=bool(visited) and all(n>=quota for n in visited),
                    coverage_complete=bool(visited) and all(n>=quota for n in visited) and denom>0 and fraction>=goal,
                    interpretation='Area-weighted sampled geometry/RGB-quality proxy; not object recall or whole-scene coverage')

    def snapshot(self):
        return dict(summary=self.summary(),patch_metadata=self.patch_metadata,
                    collision_padding=self.grid.padding,visibility_padding=self.visibility_grid.padding,
                    points=self.patches.tolist(),weights=self.weights.tolist(),
                    potential_effective=self.potential.tolist(),first_seen=self.first_seen.tolist(),
                    first_effective=self.first_effective.tolist(),stations=copy.deepcopy(self.stations))

    def verification_allowed(self):
        if self.current_station is None: return False
        s=self.stations[self.current_station]
        return sum(v>=0 for v in s['view_first_observed'])>=self.options.get('views_per_visit',3)

    def explore(self,camera,memory,planner,remaining_distance=float('inf'),force_translate=False):
        self.initialize(camera)
        if self.summary()['coverage_complete']:
            return None,None,dict(reason='sampled_coverage_goal_reached')
        quota=self.options.get('views_per_visit',3)
        unseen=self.first_seen<0;unresolved=self.first_effective<0
        options=[]
        minimum=self.config.get('min_explore_translation',1.5)
        if force_translate:
            remaining=[np.linalg.norm(np.array(s['position'])-camera.position) for s in self.stations
                       if any(v<0 for v in s['view_first_observed'])]
            if remaining and max(remaining)<minimum: minimum=1e-4
        for i,s in enumerate(self.stations):
            counts=sum(v>=0 for v in s['view_first_observed'])
            available=[j for j,v in enumerate(s['view_first_observed']) if v<0]
            if not available: continue
            distance=float(np.linalg.norm(np.array(s['position'])-camera.position))
            if distance>remaining_distance+1e-8: continue
            if force_translate and distance<minimum: continue
            candidates=[]
            for j in available:
                seen,effective=self.masks[i][j]
                gain=float(self.weights[effective&unresolved].sum())+float(self.weights[seen&unseen].sum())*.15
                turn=math.acos(np.clip(camera.c2w[:3,2]@self.views[i][j].c2w[:3,2],-1,1))
                candidates.append((gain-.025*turn,j,gain))
            _,j,gain=max(candidates)
            phase=counts//quota
            # Complete a short burst on arrival; after it, visit other regions.
            locked=(i==self.current_station and distance<1e-4 and counts-self.visit_start<quota)
            priority=(-1 if locked else phase)
            semantic=sum(min(1,o['candidate_count'])*math.exp(-np.linalg.norm(np.array(o['position'])-s['position'])) for o in memory.observations[-8:])
            score=(gain+.1+min(.2,semantic))/(1+.35*distance)
            options.append((priority,-score,i,j,gain,distance))
        failures=[]
        for _,_,i,j,gain,distance in sorted(options):
            station=self.stations[i]
            # Retry routes from a different origin, but never mark a sampled
            # reachable station geometrically unreachable on optimizer failure.
            origin=np.round(camera.position/self.grid.voxel).astype(int).tolist()
            if any(f['origin']==origin for f in station['planning_failures']): continue
            try:
                if distance<1e-6:
                    path=np.array([camera.position]);self.grid.validate_path(path)
                    details=dict(backend='stationary_scan',fallback_used=False)
                else:
                    path=planner.plan(camera.position,station['position']);details=dict(planner.last_metadata)
                length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
                if length>remaining_distance+1e-8:
                    failures.append(dict(station_id=i,reason='route_exceeds_remaining_path_budget',path_length=length));continue
                if i!=self.current_station:
                    self.current_station=i;self.visit_start=sum(v>=0 for v in station['view_first_observed'])
                return self.views[i][j],path,dict(reason='persistent_station_surface_coverage',station_id=i,station_view=j,
                        new_surface_gain=gain,path_cost=length,force_translate=force_translate,
                        planner_actual_backend=details.get('backend',planner.backend),
                        planner_fallback_used=details.get('fallback_used',False),
                        planner_primary_error=details.get('primary_error'),planning_failures=failures)
            except PlanningError as exc:
                failure=dict(station_id=i,origin=origin,error=str(exc));station['planning_failures'].append(failure);failures.append(failure)
        exhausted=all(all(v>=0 for v in s['view_first_observed']) for s in self.stations)
        return None,None,dict(reason='coverage_views_exhausted' if exhausted else 'no_feasible_coverage_route_within_budget',planning_failures=failures)
