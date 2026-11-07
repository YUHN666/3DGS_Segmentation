"""A fixed 16-position x 8-yaw schedule, matching gs_sam2_instances.

Only camera geometry is imported. No SAM masks, scene labels, source RGB/depth,
semantic coordinates or candidate previews enter the schedule or recognition.
"""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from .camera import look_at
from .navigation import PlanningError
from .policy import ViewPolicy


def expand_grid_cameras(layout):
    """Same horizontal direction/up convention as reference common.parse_cameras."""
    up=np.asarray(layout['up'],float)
    if up.shape!=(3,) or not np.isfinite(up).all() or np.linalg.norm(up)<1e-10:
        raise ValueError('Grid up must be a finite nonzero xyz')
    up=up/np.linalg.norm(up)
    forward=np.asarray(layout.get('forward',[0,0,1]),float)
    forward=forward-up*(forward@up)
    if not np.isfinite(forward).all() or np.linalg.norm(forward)<1e-10: raise ValueError('Invalid horizontal forward')
    forward/=np.linalg.norm(forward);side=np.cross(up,forward)
    yaw=np.asarray(layout['yaw_degrees'],float)
    if yaw.shape!=(8,) or not np.allclose(yaw,np.arange(8)*45):
        raise ValueError('grid128 requires yaw_degrees [0,45,...,315]')
    positions=layout['camera_positions']
    if len(positions)!=16 or len({s['name'] for s in positions})!=16:
        raise ValueError('grid128 requires 16 uniquely named fixed positions')
    width,height=int(layout['width']),int(layout['height']);K=np.asarray(layout['K'],float)
    output=[]
    for station in positions:
        point=np.asarray(station['position'],float)
        if point.shape!=(3,) or not np.isfinite(point).all(): raise ValueError('Invalid grid position')
        views=[]
        for angle in yaw:
            direction=np.cos(np.radians(angle))*forward+np.sin(np.radians(angle))*side
            views.append(look_at(point,direction,up,K,width,height))
        output.append(views)
    return output


class FixedGridScan(ViewPolicy):
    scheduled_observations_only=True
    adaptive_verification=False

    def __init__(self,geometry,grid,config,camera_config):
        super().__init__(geometry,grid,config,camera_config)
        self.options=config['grid128']
        path=Path(self.options['positions_file'])
        self.layout=json.loads(path.read_text())
        self.reference=dict(positions_file=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            preparation=self.layout.get('preparation'))
        self.views=expand_grid_cameras(self.layout)
        self.observed=np.full((16,8),-1,int)
        # Fixed serpentine position order avoids jumping across a completed row.
        order=list(range(4))+list(range(7,3,-1))+list(range(8,12))+list(range(15,11,-1))
        if self.options.get('position_order','serpentine')=='reference': order=list(range(16))
        elif self.options.get('position_order','serpentine')!='serpentine': raise ValueError('Invalid position_order')
        self.order=order;self.schedule=[(i,j) for i in order for j in range(8)]
        self.initialized=False

    def initialize(self,camera):
        if self.initialized: return
        first=self.views[0][0]
        if camera.width!=first.width or camera.height!=first.height or not np.allclose(camera.K,first.K):
            raise ValueError('Fixed grid intrinsics must match the reference layout; edit camera config/layout together')
        root=self.grid.labels[self.grid.require_free(camera.position,'start')]
        positions=[]
        for i,views in enumerate(self.views):
            point=views[0].position
            idx=self.grid.require_free(point,f'grid position {self.layout["camera_positions"][i]["name"]}')
            if self.grid.labels[idx]!=root: raise PlanningError(f'Grid position {i} is unreachable from start')
            if any(np.linalg.norm(point-p)<1e-6 for p in positions): raise ValueError('Duplicate fixed positions')
            positions.append(point)
        self.initialized=True

    def next_pair(self):
        return next(((i,j) for i,j in self.schedule if self.observed[i,j]<0),None)

    def is_scheduled_view(self,camera):
        pair=self.next_pair()
        if pair is None: return False
        target=self.views[pair[0]][pair[1]]
        return (camera.width==target.width and camera.height==target.height and
                np.allclose(camera.c2w,target.c2w,atol=1e-6,rtol=0) and np.allclose(camera.K,target.K))

    @property
    def scan_complete(self): return bool((self.observed>=0).all())

    def record_arrived(self,camera,rgb):
        if not self.is_scheduled_view(camera): raise RuntimeError('Detection attempted outside the next scheduled grid view')
        i,j=self.next_pair()
        result=super().record_arrived(camera,rgb)
        self.observed[i,j]=len(self.history)-1
        return dict(result,grid_scan=dict(position_index=i,position_name=self.layout['camera_positions'][i]['name'],
                                         yaw_degrees=j*45,completed=int((self.observed>=0).sum()),total=128))

    def scan_snapshot(self):
        return dict(strategy='grid128',views_total=128,views_observed=int((self.observed>=0).sum()),
                    positions_total=16,positions_complete=int((self.observed>=0).all(axis=1).sum()),
                    scan_complete=self.scan_complete,reference=self.reference,
                    position_order=self.order,
                    stations=[dict(name=s['name'],position=s['position'],view_observation_ids=self.observed[i].tolist())
                              for i,s in enumerate(self.layout['camera_positions'])],
                    yaw_degrees=self.layout['yaw_degrees'],
                    cameras=[dict(name=f'{self.layout["camera_positions"][i]["name"]}_y{j:02d}',**self.views[i][j].as_dict())
                             for i,j in self.schedule])

    def explore(self,camera,memory,planner,remaining_distance=float('inf'),force_translate=False):
        self.initialize(camera)
        pair=self.next_pair()
        if pair is None: return None,None,dict(reason='fixed_grid_128_completed')
        i,j=pair;target=self.views[i][j]
        if np.linalg.norm(camera.position-target.position)<1e-8:
            path=np.array([camera.position]);self.grid.validate_path(path)
            details=dict(backend='stationary_grid_scan',fallback_used=False)
        else:
            path=planner.plan(camera.position,target.position);details=dict(planner.last_metadata)
        length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
        if length>remaining_distance+1e-8:
            return None,None,dict(reason='grid_route_exceeds_remaining_path_budget',required_path_length=length)
        return target,path,dict(reason='fixed_grid_scheduled_view',station_id=i,position_name=self.layout['camera_positions'][i]['name'],
                               grid_view=j,yaw_degrees=j*45,scan_index=int((self.observed>=0).sum()),scan_total=128,
                               path_cost=length,planner_actual_backend=details.get('backend',planner.backend),
                               planner_fallback_used=details.get('fallback_used',False),planner_primary_error=details.get('primary_error'))
