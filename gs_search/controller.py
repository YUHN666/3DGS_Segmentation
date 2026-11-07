import time
import traceback
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .camera import Camera
from .memory import ObservationMemory


def evidence_area(evidence):
    x0,y0,x1,y1=evidence.detection.box
    return max(0,x1-x0)*max(0,y1-y0)/(evidence.camera.width*evidence.camera.height)


class CameraEnvironment:
    """Actuator + actual-arrival observation gate. Policy never owns renderer."""
    def __init__(self, camera, grid, renderer):
        self.camera, self.grid, self._renderer = camera, grid, renderer
        self.grid.require_free(camera.position, "start")
        self.distance = 0.
        self._current_rgb = None
        self.trajectory = [dict(position=camera.position.tolist(), camera=camera.as_dict(), path_length=0., action="start")]

    def move(self, camera):
        if not self.grid.segment_free(self.camera.position, camera.position):
            raise RuntimeError("Camera actuator rejected continuous segment collision")
        self.distance += float(np.linalg.norm(self.camera.position-camera.position))
        self.camera = camera
        self._current_rgb = None
        self.trajectory.append(dict(position=camera.position.tolist(), camera=camera.as_dict(),
                                    path_length=self.distance, action="move"))

    def observe(self):
        self.grid.require_free(self.camera.position, "observation pose")
        if self._current_rgb is None:
            self._current_rgb = self._renderer.render(self.camera)
        return self._current_rgb


class SearchController:
    def __init__(self, environment, planner, grounder, verifier, policy, artifacts, config):
        self.env, self.planner, self.grounder, self.verifier = environment, planner, grounder, verifier
        self.policy, self.artifacts, self.config = policy, artifacts, config
        self.memory = ObservationMemory(config["association"])
        self.state, self.reason, self.observations = "EXPLORE", "running", 0
        self.active = None
        self.search_mode=config['search'].get('mode','search_first')
        if self.search_mode not in ['search_first','coverage_first']:
            raise ValueError('search.mode must be search_first/coverage_first')
        self.confirmed_ids=[]
        if hasattr(policy,'initialize'):
            policy.initialize(environment.camera)
            if hasattr(policy,'snapshot'): artifacts.event('coverage_initialized',coverage=policy.snapshot())
            if hasattr(policy,'scan_snapshot'): artifacts.event('grid_scan_initialized',scan=policy.scan_snapshot())
        self.started = time.perf_counter()
        self.times = dict(render=0., grounding=0., association_and_review=0., coverage=0., planning_and_scoring=0., movement=0.)
        self.visual = getattr(artifacts, 'visualization', None)
        self.verify_streak = 0
        self.verification_observations=0
        self.force_translate = False

    def transition(self, state, reason, **data):
        self.artifacts.event("state_transition", previous=self.state, state=state, reason=reason, **data)
        self.state = state

    def budget_reason(self):
        if getattr(self.policy,'scan_complete',False): return 'fixed_grid_128_completed'
        b = self.config["search"]
        if self.observations >= b["max_observations"]:
            return "observation_budget_exhausted"
        if time.perf_counter()-self.started >= b["max_seconds"]:
            return "time_budget_exhausted"
        if self.env.distance >= b["max_path_length"]-1e-8:
            return "path_budget_exhausted"
        return None

    def observe(self, allow_verify=True):
        was_verifying=self.state=='VERIFY'
        if was_verifying: self.verification_observations+=1
        t = time.perf_counter()
        rgb = self.env.observe()
        self.times["render"] += time.perf_counter()-t
        if self.visual is not None and not self.visual.frames:
            self.visual.capture(rgb,self.env.camera,self.state,self.env.distance)
        obs = self.observations
        # Preserve actual RGB before inference, including in model failure cases.
        from PIL import Image
        Image.fromarray(rgb).save(self.artifacts.path/"rgb"/f"{obs:04d}.png")
        self.artifacts.event("rgb_arrived", observation_id=obs, camera=self.env.camera.as_dict(),
                             rgb=f"rgb/{obs:04d}.png")
        self.observations += 1
        self.artifacts.progress(f'检测关键帧 {obs} · {self.state}')
        t = time.perf_counter()
        try:
            detections = self.grounder.detect(rgb, self.config["target"]["description"])
        finally:
            if getattr(self.grounder,'last_response',None) is not None:
                from .artifacts import write_json
                directory=self.artifacts.path/'grounding';directory.mkdir(exist_ok=True)
                write_json(directory/f'{obs:04d}.json',self.grounder.last_response)
        self.times["grounding"] += time.perf_counter()-t
        t = time.perf_counter()
        if self.search_mode=='coverage_first' and self.confirmed_ids:
            # The requested target evidence is already saved. Keep raw new
            # detections, but do not accumulate unrelated instance candidates
            # while finishing the geometric coverage task.
            self.memory.observations.append(dict(id=obs,position=self.env.camera.position.tolist(),candidate_count=0))
            touched,associations=[],[]
        else:
            touched, associations = self.memory.ingest(obs, self.env.camera, rgb, detections)
        found = None
        for candidate in touched:
            if self.search_mode=='coverage_first' and self.confirmed_ids:
                break
            if self.memory.confirm(candidate, self.verifier, self.config["target"]["description"],
                                   self.config["target"].get("require_full_description", False)):
                found = candidate
                break
        for detection in detections:
            for candidate in touched:
                if any(e.observation_id==obs and e.detection is detection for e in candidate.sightings):
                    detection.candidate_id=candidate.id
                    detection.review_status=candidate.status
                    break
        self.memory.observations[-1]['candidate_count']=sum(
            not d.review_status.startswith('rejected') for d in detections)
        self.times["association_and_review"] += time.perf_counter()-t
        t = time.perf_counter()
        coverage = self.policy.record_arrived(self.env.camera, rgb)
        self.times["coverage"] += time.perf_counter()-t
        self.artifacts.observation(obs, rgb, self.env.camera, detections, [c.id for c in touched],
                                   dict(state=self.state, coverage=coverage, associations=associations,
                                        path_length=self.env.distance))
        self.env.trajectory.append(dict(position=self.env.camera.position.tolist(),
                                        camera=self.env.camera.as_dict(), path_length=self.env.distance,
                                        action="observe", observation_id=obs))
        if found:
            if found.id not in self.confirmed_ids:
                self.confirmed_ids.append(found.id)
                self.artifacts.event('target_confirmed',candidate_id=found.id,continue_coverage=self.search_mode=='coverage_first')
            if self.search_mode=='search_first':
                self.active = found
                self.transition("FOUND", "translated_multiview_rgb_evidence", candidate_id=found.id)
                return True
        if self.search_mode=='coverage_first' and self.confirmed_ids:
            self.active=None
            self.verify_streak=0
            self.force_translate=False
            if self.state!='EXPLORE': self.transition('EXPLORE','target_saved_continue_coverage')
            return False
        else:
            if was_verifying: self.verify_streak+=1
            if self.verify_streak>=self.config['search'].get('max_consecutive_verify_observations',4):
                self.force_translate=True
                self.active=None
                self.transition('EXPLORE','local_verification_budget_requires_spatial_exploration')
                return False
            eligible = [c for c in self.memory.candidates if c.verify_attempts < self.config["search"].get("max_verify_attempts", 3)
                        and not c.status.startswith(("confirmed","rejected"))
                        and self.observations-c.sightings[-1].observation_id<=self.config['search'].get('candidate_max_age',3)
                        and np.linalg.norm(c.sightings[0].camera.position-self.env.camera.position)
                            <=self.config['search'].get('candidate_max_anchor_distance',1.5)]
            station_ready=(self.search_mode!='coverage_first' or was_verifying or
                           not hasattr(self.policy,'verification_allowed') or self.policy.verification_allowed())
            verification_budget=(self.search_mode!='coverage_first' or self.verification_observations<
                                 self.config['search'].get('max_verification_observations',24))
            if (not allow_verify or self.force_translate or not station_ready or not verification_budget or
                    not getattr(self.policy,'adaptive_verification',True)):
                self.active=None
                if self.state!='EXPLORE':
                    self.transition('EXPLORE','verification_deferred_until_exploration_progress')
                return False
            if self.active not in eligible:
                self.active = max(eligible,key=lambda c:(len(c.sightings),max(map(evidence_area,c.sightings))),default=None)
            if self.active:
                if self.state != "VERIFY":
                    self.transition("VERIFY", "candidate_requires_translated_rgb_review", candidate_id=self.active.id)
            elif self.state == "VERIFY":
                self.force_translate=True
                self.transition("EXPLORE", "verification_attempts_exhausted_candidates_remain_pending")
        return bool(touched)

    def run(self):
        error = None
        try:
            if self.budget_reason():
                self.reason = self.budget_reason()
            else:
                if not getattr(self.policy,'scheduled_observations_only',False) or self.policy.is_scheduled_view(self.env.camera):
                    self.observe()
            while self.state != "FOUND" and not self.budget_reason():
                remaining = self.config["search"]["max_path_length"]-self.env.distance
                t = time.perf_counter()
                if self.visual is not None:
                    self.visual.set_phase('规划复核视角' if self.state=='VERIFY' else '选择并规划探索视角')
                if self.state == "VERIFY" and self.active:
                    view, path, decision = self.policy.verify(self.env.camera, self.active, self.planner, remaining)
                    self.active.verify_attempts += 1
                    if view is None:
                        self.active.verify_attempts = self.config["search"].get("max_verify_attempts", 3)
                        self.artifacts.event("view_decision", state=self.state, **decision)
                        self.active = None
                        self.force_translate=True
                        self.transition("EXPLORE", "verification_route_unavailable_evidence_kept_pending")
                        self.times["planning_and_scoring"] += time.perf_counter()-t
                        continue
                else:
                    view, path, decision = self.policy.explore(self.env.camera, self.memory, self.planner, remaining,
                                                              force_translate=self.force_translate)
                self.times["planning_and_scoring"] += time.perf_counter()-t
                self.artifacts.event("view_decision", state=self.state, view=view.as_dict() if view else None,
                                     path=path.tolist() if path is not None else None, **decision)
                if view is None:
                    self.reason = decision["reason"]
                    break
                self.env.grid.validate_path(path)
                initial = self.env.camera
                leg_start_distance=self.env.distance
                leg_length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
                station_transit=self.search_mode=='coverage_first' and decision.get('station_id') is not None and leg_length>1e-6
                committed_exploration=bool(decision.get('force_translate')) or station_transit
                commitment=leg_length if station_transit else min(self.config['search'].get('min_explore_progress',1.5),leg_length)
                last_observe_distance = self.env.distance
                interrupted = False
                if self.visual is not None:
                    from .motion import along_path
                    steps = along_path(initial,view,path,self.visual.options)
                    self.visual.set_phase('相机移动与实时渲染')
                else:
                    steps = self._basic_steps(initial,view,path)
                # Render only after the actuator has reached each sample.
                for step in steps:
                    if self.budget_reason():
                        interrupted = True
                        break
                    step_started = time.perf_counter()
                    if self.visual is not None:
                        self.visual.set_phase('相机移动与实时渲染')
                    t = time.perf_counter()
                    self.env.move(step.camera)
                    self.times["movement"] += time.perf_counter()-t
                    if self.visual is not None and self.visual.can_capture:
                        t = time.perf_counter()
                        rgb = self.env.observe()
                        self.times['render'] += time.perf_counter()-t
                        self.visual.capture(rgb,self.env.camera,self.state,self.env.distance)
                    arrived = step.arrived
                    if arrived or (not getattr(self.policy,'scheduled_observations_only',False) and
                                   self.env.distance-last_observe_distance >= self.config["search"].get("observe_every_distance", .6)):
                        previous_state = self.state
                        progress=self.env.distance-leg_start_distance
                        if committed_exploration and (progress>=commitment-1e-6 or arrived):
                            self.force_translate=False
                            self.verify_streak=0
                            committed_exploration=False
                            self.artifacts.event('exploration_commitment_completed',distance=progress)
                        new_clue = self.observe(allow_verify=not committed_exploration)
                        last_observe_distance = self.env.distance
                        if self.state == "FOUND":
                            break
                        if new_clue and previous_state == "EXPLORE" and self.state=='VERIFY':
                            self.artifacts.event("path_interrupted", reason="new_rgb_candidate", observation_id=self.observations-1)
                            interrupted = True
                            break
                    if self.visual is not None and self.visual.options.get('realtime',False):
                        # Pace movement when fast enough; slow inference/rendering
                        # naturally lowers the live FPS rather than skipping checks.
                        delay = step.duration-(time.perf_counter()-step_started)
                        if delay > 0:
                            time.sleep(min(delay,max(0,self.config['search']['max_seconds']-(time.perf_counter()-self.started))))
                if self.state == "FOUND":
                    break
                if interrupted and self.budget_reason():
                    break
            if self.search_mode=='coverage_first' and self.confirmed_ids:
                self.reason=self.budget_reason() or self.reason
                self.transition('FOUND','target_evidence_saved_coverage_stopped',coverage_stop_reason=self.reason)
            elif self.state != "FOUND":
                self.reason = self.budget_reason() or self.reason
                self.transition("INCOMPLETE", self.reason)
            else:
                self.reason = "translated_multiview_rgb_evidence"
        except (Exception, KeyboardInterrupt) as exc:
            error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
            self.reason = "user_interrupt" if isinstance(exc,KeyboardInterrupt) else "runtime_error"
            self.transition("INCOMPLETE", self.reason, error=error)
        confirmed=[c for c in self.memory.candidates if c.id in self.confirmed_ids]
        best=confirmed[0] if confirmed else self.active
        coverage=self.policy.summary() if hasattr(self.policy,'summary') else None
        if hasattr(self.policy,'snapshot'):
            from .artifacts import write_json
            write_json(self.artifacts.path/'coverage.json',self.policy.snapshot())
        scan=self.policy.scan_snapshot() if hasattr(self.policy,'scan_snapshot') else None
        if scan is not None:
            from .artifacts import write_json
            write_json(self.artifacts.path/'grid_scan.json',scan)
        report = dict(status=self.state, stop_reason=self.reason,
                      interpretation=("已保存多视图确认候选；覆盖是否完成请单独查看 coverage_status" if confirmed
                                      else "尚未找到；不能断言目标不存在"),
                      target=self.config["target"], candidates=[c.summary() for c in self.memory.candidates],
                      best_observation_pose=(max(best.sightings, key=evidence_area).camera.as_dict()
                                             if best else None),
                      search_mode=self.search_mode,target_found=bool(self.confirmed_ids),confirmed_candidate_ids=self.confirmed_ids,
                      verification_observations=self.verification_observations,
                      grid_scan=scan,scan_status=('COMPLETE' if scan['scan_complete'] else 'PARTIAL') if scan else 'NOT_APPLICABLE',
                      coverage=coverage,coverage_status=('COMPLETE' if coverage['coverage_complete'] else 'PARTIAL') if coverage else 'NOT_MEASURED',
                      path_length=self.env.distance, observations=self.observations,
                      render_count=self.env._renderer.calls, vlm_grounding_calls=self.grounder.calls,
                      display_only_render_count=max(0,self.env._renderer.calls-self.observations),
                      vlm_calls=self.verifier.calls, timing_seconds=self.times,
                      vlm_model_info=getattr(self.verifier,'model_info',None),
                      vlm_inference_metrics=getattr(self.verifier,'metrics',[]),
                      initialization_seconds=next((e["seconds"] for e in self.artifacts.events if e["kind"] == "initialized"), None),
                      backends=dict(renderer=self.env._renderer.backend, planner=self.planner.backend,
                                    grounding=self.grounder.backend, verifier=self.verifier.mode),
                      executed_planner_backends=[e.get('planner_actual_backend') for e in self.artifacts.events
                                                 if e['kind']=='view_decision' and e.get('path') is not None],
                      error=error, target_xyz=None,
                      limits=["Scores are not calibrated probabilities", "Coverage is a view-quality proxy, not exhaustive object coverage",
                              "No target 3-D localization; no depth provided to perception",
                              "Cross-view association is conservative; textureless/ambiguous objects remain pending",
                              "VLM disabled: attributes/relations not verified" if self.verifier.mode == "disabled" else
                              "VLM judgments are model evidence, not ground truth"])
        self.artifacts.finish(report, self.env.trajectory)
        return report

    @staticmethod
    def _basic_steps(initial,view,path):
        from .motion import MotionStep
        if len(path)==1:
            yield MotionStep(view,0.,True)
            return
        slerp=Slerp([0,1],Rotation.from_matrix(np.stack([initial.c2w[:3,:3],view.c2w[:3,:3]])))
        for i,position in enumerate(path[1:],1):
            pose=np.eye(4)
            pose[:3,:3],pose[:3,3]=slerp(i/(len(path)-1)).as_matrix(),position
            yield MotionStep(Camera(pose,view.K,view.width,view.height),0.,i==len(path)-1)
