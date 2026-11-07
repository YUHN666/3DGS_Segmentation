"""Control-only tests use explicitly named scripted RGB/detection doubles."""
import json
from pathlib import Path
import inspect
import numpy as np
from gs_search.artifacts import RunArtifacts
from gs_search.camera import Camera
from gs_search.controller import CameraEnvironment, SearchController
from gs_search.navigation import OccupancyMap, AStarPlanner
from gs_search.scene import Geometry
from gs_search.perception import Detection, QwenVerifier
from gs_search.vlm_grounding import VLMGrounder
from gs_search.policy import ViewPolicy


class ScriptedRGB:
    backend = 'TEST_DOUBLE_RGB'
    def __init__(self):
        self.calls = 0
        self.rng = np.random.default_rng(3)
    def render(self, camera):
        self.calls += 1
        return self.rng.integers(0,256,(camera.height,camera.width,3),dtype=np.uint8)


class ScriptedGrounder:
    backend = 'TEST_DOUBLE_DETECTOR'
    def __init__(self, empty=False):
        self.calls, self.empty = 0, empty
    def detect(self, rgb, category):
        assert isinstance(rgb,np.ndarray) and rgb.ndim == 3
        self.calls += 1
        return [] if self.empty else [Detection([40,30,120,110],.8,'chair')]


def setup(tmp_path, empty):
    geometry = Geometry(np.empty((0,3)),np.empty((0,3)),np.empty((0,4)),np.empty(0))
    config = dict(bounds=[[-3,-2,-3],[3,2,3]],voxel_size=.2,camera_radius=.04,margin=.02,move_step=.1)
    grid = OccupancyMap(geometry,config)
    pose = np.eye(4)
    pose[:3,3] = [.1,.1,.1]
    cam = Camera(pose,[[150,0,80],[0,150,60],[0,0,1]],160,120)
    c = dict(target=dict(description='chair',category='chair'), association={},
             search=dict(max_observations=3,max_seconds=60,max_path_length=2,observe_every_distance=.7,max_verify_attempts=2))
    env = CameraEnvironment(cam,grid,ScriptedRGB())
    policy = ViewPolicy(geometry,grid,dict(position_samples=2,yaw_samples=3,coverage_samples=0,verify_baseline=.5),dict(world_up=[0,-1,0]))
    run = RunArtifacts(tmp_path,c,'CONTROL_TEST')
    return SearchController(env,AStarPlanner(grid),ScriptedGrounder(empty),QwenVerifier({'mode':'disabled'}),policy,run,c)


def test_empty_detection_budget_is_incomplete(tmp_path):
    controller = setup(tmp_path,True)
    r = controller.run()
    assert r['status'] == 'INCOMPLETE' and not r['candidates']
    assert r['observations'] == 3 and r['stop_reason'] == 'observation_budget_exhausted'
    assert r['error'] is None
    assert (controller.artifacts.path/'index.html').exists()


def test_moving_search_uses_shared_vlm_and_scoreless_proposals(tmp_path):
    controller=setup(tmp_path,False)
    verifier=QwenVerifier({'mode':'disabled','category_prompt':'chair'})
    verifier.mode='TEST_DOUBLE'
    def generate(images,prompt):
        assert len(images)==1 and images[0].dtype==np.uint8
        verifier.calls+=1
        return '{"objects":[{"label":"TEST_DOUBLE chair","bbox_2d":[250,250,750,900]}]}'
    verifier._generate=generate
    controller.verifier=verifier;controller.grounder=VLMGrounder(verifier)
    report=controller.run()
    assert report['error'] is None and report['observations']==report['vlm_grounding_calls']==3
    assert report['vlm_calls']==3 and report['status']=='INCOMPLETE'
    assert all(s.detection.score is None for c in controller.memory.candidates for s in c.sightings)


def test_candidate_changes_position_and_remains_pending_without_rgb_identity(tmp_path):
    controller = setup(tmp_path,False)
    r = controller.run()
    assert r['error'] is None
    assert r['status'] == 'INCOMPLETE'
    obs = [e for e in controller.artifacts.events if e['kind']=='observation']
    positions = [np.array(e['camera']['camera_to_world'])[:3,3] for e in obs]
    assert len(positions) == 3
    assert np.linalg.norm(positions[1]-positions[0]) >= .3
    assert any(e.get('state')=='VERIFY' for e in controller.artifacts.events if e['kind']=='state_transition')
    assert all(not c['status'].startswith('confirmed') for c in r['candidates'])


def test_policy_has_no_renderer_and_detector_has_no_geometry_arguments():
    assert list(inspect.signature(VLMGrounder.detect).parameters) == ['self','rgb','target_text']
    assert 'renderer' not in inspect.signature(ViewPolicy.__init__).parameters
    assert 'scene' not in inspect.signature(VLMGrounder.__init__).parameters


def test_found_requires_actual_rgb_correspondences_from_translated_views(tmp_path):
    import cv2
    class SyntheticTexturedPlane:
        backend = 'TEST_SYNTHETIC_PLANAR_RGB'
        def __init__(self):
            self.calls = 0
            rng = np.random.default_rng(21)
            self.texture = rng.integers(0,256,(480,640,3),dtype=np.uint8)
            for _ in range(140):
                xy = tuple(rng.integers([0,0],[640,480]).tolist())
                cv2.circle(self.texture,xy,int(rng.integers(3,15)),tuple(rng.integers(0,255,3).tolist()),-1)
        def render(self, camera):
            self.calls += 1
            corners = np.array([[-1.6,-1,3],[1.6,-1,3],[1.6,1,3],[-1.6,1,3]])
            q = (corners-camera.position) @ camera.c2w[:3,:3]
            uv = (q @ camera.K.T)
            uv = (uv[:,:2]/uv[:,2:]).astype(np.float32)
            H = cv2.getPerspectiveTransform(np.array([[0,0],[639,0],[639,479],[0,479]],np.float32),uv)
            return cv2.warpPerspective(self.texture,H,(camera.width,camera.height))
    class PlaneCategoryDouble(ScriptedGrounder):
        def detect(self,rgb,category):
            self.calls += 1
            return [Detection([2,2,158,118],.9,'chair')]
    controller = setup(tmp_path,False)
    controller.env._renderer = SyntheticTexturedPlane()
    controller.grounder = PlaneCategoryDouble()
    controller.config['association']['min_feature_box_fraction'] = .05
    controller.config['association']['max_box_area_fraction'] = .98
    controller.config['association']['box_border_margin'] = 1
    r = controller.run()
    assert r['error'] is None
    assert r['status'] == 'FOUND', r['candidates']
    assert r['observations'] == 2 and r['path_length'] >= .3
    assert r['candidates'][0]['association_links'][0]['inliers'] >= 10
    # Even geometrically associated category evidence cannot silently satisfy
    # a full Chinese attribute description when the VLM is disabled.
    from gs_search.memory import Candidate
    source = next(c for c in controller.memory.candidates if c.status.startswith('confirmed'))
    fresh = Candidate('full_description_check',source.sightings.copy(),source.links.copy())
    assert not controller.memory.confirm(fresh,QwenVerifier({'mode':'disabled'}),'绿色的椅子',True)
    assert fresh.status == 'pending_full_description_requires_vlm'
