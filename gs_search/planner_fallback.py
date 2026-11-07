"""Explicitly configured, independently named geometric fallback.

Used when the upstream optimized polyline fails the conservative voxel check.
The report names the hybrid backend and every chosen leg records its actual
backend. A fallback path is never described as a SplatPlan optimization.
"""
from .navigation import AStarPlanner,PlanningError


class ExplicitPlannerFallback:
    backend='splatplan_checked_seed+occupancy_astar_3d_fallback'

    def __init__(self,primary,grid):
        self.primary=primary
        self.fallback=AStarPlanner(grid)
        self.last_metadata={}

    def plan(self,start,goal):
        try:
            path=self.primary.plan(start,goal)
            self.last_metadata=dict(self.primary.last_metadata,backend=self.primary.backend,fallback_used=False)
            return path
        except PlanningError as exc:
            error=repr(exc)
            primary_details=self.primary.last_metadata
            path=self.fallback.plan(start,goal)
            self.last_metadata=dict(self.fallback.last_metadata,backend=self.fallback.backend,fallback_used=True,
                                    primary_error=error,primary_attempts=primary_details.get('attempts',[]))
            return path
