import numpy as np
from gs_search.camera import Camera
from gs_search.memory import Evidence, associate, ObservationMemory
from gs_search.perception import Detection, QwenVerifier


def camera(x):
    pose = np.eye(4)
    pose[0, 3] = x
    return Camera(pose, np.array([[400,0,200],[0,400,150],[0,0,1.]]), 400, 300)


def test_pose_verified_feature_matches_and_baseline():
    # Explicit synthetic correspondences, not real model/scene validation.
    rng = np.random.default_rng(4)
    xyz = rng.uniform([-1, -.7, 3], [1, .7, 5], (80, 3))
    desc = rng.uniform(0, 250, (80, 128)).astype(np.float32)
    def ev(cam, obs):
        q = (xyz-cam.position) @ cam.c2w[:3,:3]
        uv = q[:,:2]/q[:,2:]*400 + [200,150]
        return Evidence(obs, cam, Detection([10,10,390,290],.8,'chair'),
                        np.zeros((300,400,3),np.uint8), uv.astype(np.float32), desc, str(obs))
    a, b = ev(camera(-.3), 0), ev(camera(.3), 1)
    assert associate(a, b, {})['accepted']
    assert not associate(a, ev(camera(-.29), 2), {})['accepted']
    # Duplicate image must never be an independent view.
    b.image_hash = a.image_hash
    assert associate(a, b, {})['reason'] == 'duplicate_rgb'


def test_same_category_does_not_merge_instances_or_confirm_rotation():
    rgb = np.zeros((300,400,3),np.uint8)
    m = ObservationMemory({})
    ds = [Detection([10,10,150,180], .8, 'chair'), Detection([220,20,390,190], .8,'chair')]
    touched, _ = m.ingest(0, camera(0), rgb, ds)
    assert len(m.candidates) == 2
    assert touched[0].id != touched[1].id
    m.ingest(1, camera(.6), rgb.copy(), ds)
    assert len(m.candidates) == 4  # No features, no association, even same labels.
    for c in m.candidates:
        assert not m.confirm(c, QwenVerifier({'mode':'disabled'}), 'chair')


def test_wrong_pose_rejects_correspondences():
    rng = np.random.default_rng(12)
    desc = rng.uniform(0,200,(30,128)).astype(np.float32)
    pa = rng.uniform([80,60],[300,240],(30,2)).astype(np.float32)
    # Horizontal-baseline cameras but matches move vertically: impossible.
    a = Evidence(0,camera(0),Detection([0,0,400,300],.8,'chair'),None,pa,desc,'a')
    b = Evidence(1,camera(.6),Detection([0,0,400,300],.8,'chair'),None,pa+[0,35],desc,'b')
    assert not associate(a,b,{})['accepted']
