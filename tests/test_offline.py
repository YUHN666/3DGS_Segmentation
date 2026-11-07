"""Offline control/geometry tests. All model doubles explicitly labelled TEST."""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from plyfile import PlyData
from gs_search.camera import Camera
from gs_search.localization import backproject_mask,write_colored_cloud
from gs_search.offline import review_detection,run_offline
from gs_search.perception import Detection


def camera():
    pose=np.eye(4);pose[:3,3]=[3,4,5]
    return Camera(pose,[[10,0,8],[0,10,6],[0,0,1]],16,12)


def test_backprojection_uses_mask_pixels_expected_z_and_camera_pose(tmp_path):
    c=camera();mask=np.zeros((12,16),bool);mask[3:9,4:12]=True
    result,points,pixels=backproject_mask(mask,np.full(mask.shape,2.),np.ones(mask.shape),c,dict(min_points=3))
    assert result['status']=='LOCALIZED' and len(points)==48
    np.testing.assert_allclose(points[:,2],7.)
    q=(points-c.position)@c.c2w[:3,:3];uv=q@c.K.T;uv=uv[:,:2]/uv[:,2:]
    np.testing.assert_allclose(uv,pixels[:,[1,0]]+.5,atol=1e-8)
    np.testing.assert_allclose(result['target_xyz'],[3,4,7])
    dest=tmp_path/'target.ply';write_colored_cloud(dest,points,np.full(points.shape,128,np.uint8))
    assert len(PlyData.read(dest)['vertex'])==48


def test_invalid_alpha_depth_and_oversized_masks_keep_position_unknown():
    c=camera();mask=np.ones((12,16),bool)
    r,points,_=backproject_mask(mask,np.ones(mask.shape),np.ones(mask.shape),c,{})
    assert r['status']=='UNRESOLVED' and points is None
    mask[:]=False;mask[3:9,4:12]=True
    for depth,alpha in [(np.ones(mask.shape),np.zeros(mask.shape)),(np.full(mask.shape,np.nan),np.ones(mask.shape)),
                        (np.full(mask.shape,-1),np.ones(mask.shape))]:
        r,points,_=backproject_mask(mask,depth,alpha,c,{})
        assert points is None and r['target_xyz'] is None


def test_clipped_single_detection_goes_directly_to_rgb_only_verifier():
    class Verifier:
        def verify(self,images,text):
            assert len(images)==1 and images[0].dtype==np.uint8
            assert text=='白色垃圾桶'
            return dict(matches_target=True,decision='accept')
    rgb=np.zeros((12,16,3),np.uint8)
    result=review_detection(rgb,Detection([0,0,16,12],.9,'trash can'),'白色垃圾桶',Verifier())
    assert result['matches_target'] is True
    assert list(inspect.signature(review_detection).parameters)==['rgb','detection','target_text','verifier']


def reuse_fixture(tmp_path,monkeypatch,accept,include_candidate=True):
    monkeypatch.setattr('gs_search.artifacts.diagnostics',lambda:{'test_double':True})
    source=tmp_path/'original';(source/'rgb').mkdir(parents=True)
    rgb=np.arange(12*16*3,dtype=np.uint8).reshape(12,16,3)
    c=camera();events=[]
    config=dict(scene=dict(path=str(tmp_path/'TEST_scene.ply'),world_scale=1),
                output_root=str(tmp_path/'runs'),renderer=dict(backend='gsplat'),
                association={},
                vlm=dict(mode='local'),target=dict(description='白色垃圾桶',category='trash can'),
                segmentation={},localization=dict(mode='rendered_depth_after_vlm',min_points=3,mask_erode=0),offline={})
    (source/'config.json').write_text(json.dumps(config))
    for i in range(128):
        Image.fromarray(rgb).save(source/'rgb'/f'{i:04d}.png')
        ds=[dict(box=[0,0,16,12],score=.8,label='trash can')] if i==0 else []
        events.append(dict(kind='observation',observation_id=i,camera=c.as_dict(),rgb=f'rgb/{i:04d}.png',detections=ds))
    (source/'decisions.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    class Verifier:
        calls=0;metrics=[];model_info={'backend':'TEST_DOUBLE'}
        def __init__(self,options): self.config=options;self.grounding_requests=0
        def _generate(self,images,prompt):
            assert len(images)==1 and images[0].dtype==np.uint8
            self.calls+=1;self.grounding_requests+=1
            objects=[dict(label='TEST_DOUBLE candidate',bbox_2d=[250,250,750,750])] if include_candidate and self.grounding_requests==1 else []
            return json.dumps(dict(objects=objects))
        def verify(self,images,target):
            self.calls+=1
            return dict(matches_target=accept,decision='accept' if accept else 'reject')
    class Segmenter:
        calls=0
        def __init__(self,*_):
            assert accept,'Rejected semantics must never construct SAM2/localizer'
        def segment(self,*_):
            self.calls+=1;mask=np.zeros((12,16),bool);mask[3:9,4:12]=True
            return mask,dict(backend='TEST_DOUBLE')
    class Renderer:
        calls=0
        def __init__(self,*_):
            assert accept,'Rejected semantics must never render localization depth'
        def render_batch_rgbd(self,cameras):
            self.calls+=len(cameras)
            return [dict(rgb=rgb.copy(),expected_z=np.full((12,16),2.),alpha=np.ones((12,16))) for _ in cameras]
    monkeypatch.setattr('gs_search.perception.QwenVerifier',Verifier)
    monkeypatch.setattr('gs_search.localization.SAM2BoxSegmenter',Segmenter)
    monkeypatch.setattr('gs_search.render_rgbd.OfflineRGBDRenderer',Renderer)
    monkeypatch.setattr('gs_search.scene.load_scene',lambda _:SimpleNamespace(metadata={'backend':'TEST_DOUBLE'}))
    return config,source


def test_reused_proposal_single_view_vlm_accept_can_localize_without_association(tmp_path,monkeypatch):
    config,source=reuse_fixture(tmp_path,monkeypatch,True)
    root,r=run_offline(config,source)
    assert r['error'] is None and r['status']=='FOUND'
    assert r['vlm_calls']==129 and r['sam2_calls']==1 and r['render_count']==1
    assert r['vlm_grounding_calls']==128 and r['reused_rgb_observations']==128
    assert len(r['localized_candidate_ids'])==1
    np.testing.assert_allclose(r['candidates'][0]['target_xyz'],[3,4,7])
    assert (root/r['candidates'][0]['localization']['pointcloud']).is_file()
    assert r['navigation_executed'] is False and not r['instances_merged']
    assert len(list((source/'rgb').glob('*.png')))==128


def test_vlm_reject_skips_geometry_and_never_reports_found(tmp_path,monkeypatch):
    config,source=reuse_fixture(tmp_path,monkeypatch,False)
    config['gaussian_projection']={'enabled':True}
    def forbidden(*_): raise AssertionError('Rejected RGB semantics must never label Gaussian rows')
    monkeypatch.setattr('gs_search.gaussian_projection.export_gaussian_ply',forbidden)
    _,r=run_offline(config,source)
    assert r['error'] is None and r['status']=='INCOMPLETE'
    assert r['vlm_calls']==129 and r['sam2_calls']==0 and r['render_count']==0
    assert r['candidates'][0]['status']=='rejected_vlm'
    assert not r['localized_candidate_ids']


def test_offline_exports_gaussian_ply_only_after_accepted_mask_and_depth(tmp_path,monkeypatch):
    config,source=reuse_fixture(tmp_path,monkeypatch,True)
    config['gaussian_projection']={'enabled':True};calls=[]
    def export(root,config,candidates):
        assert candidates[0]['semantic_review']['matches_target'] is True
        assert (root/candidates[0]['localization']['mask']).is_file()
        assert (root/'geometry/0000.npz').is_file()
        calls.append(candidates[0]['id'])
        return dict(status='TEST_DOUBLE',selected_gaussians=1)
    monkeypatch.setattr('gs_search.gaussian_projection.export_gaussian_ply',export)
    _,r=run_offline(config,source)
    assert calls==['candidate_0000'] and r['gaussian_projection']['selected_gaussians']==1
    assert r['vlm_calls']==129 and r['processing_complete']


def test_offline_ply_failure_keeps_semantic_evidence_and_reports_incomplete_processing(tmp_path,monkeypatch):
    config,source=reuse_fixture(tmp_path,monkeypatch,True)
    config['gaussian_projection']={'enabled':True}
    def fail(*_): raise MemoryError('TEST_DOUBLE export disk budget')
    monkeypatch.setattr('gs_search.gaussian_projection.export_gaussian_ply',fail)
    root,r=run_offline(config,source)
    assert r['target_found'] and not r['processing_complete']
    assert 'TEST_DOUBLE' in r['gaussian_projection_error']['message']
    assert (root/'candidates.json').is_file()


def test_empty_grounding_needs_no_review_or_geometry_models(tmp_path,monkeypatch):
    config,source=reuse_fixture(tmp_path,monkeypatch,False,include_candidate=False)
    events=[json.loads(s) for s in (source/'decisions.jsonl').read_text().splitlines()]
    for e in events:e['detections']=[]
    (source/'decisions.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    _,r=run_offline(config,source)
    assert r['status']=='INCOMPLETE' and r['error'] is None and r['vlm_calls']==128
    assert r['candidates']==[]


def test_reused_rgb_geometry_mismatch_cannot_produce_target_xyz(tmp_path,monkeypatch):
    config,source=reuse_fixture(tmp_path,monkeypatch,True)
    class WrongRenderer:
        calls=0
        def __init__(self,*_):pass
        def render_batch_rgbd(self,cameras):
            self.calls+=1
            return [dict(rgb=np.zeros((12,16,3),np.uint8),expected_z=np.full((12,16),2.),alpha=np.ones((12,16)))]
    monkeypatch.setattr('gs_search.render_rgbd.OfflineRGBDRenderer',WrongRenderer)
    _,r=run_offline(config,source)
    assert r['target_found'] and not r['localized_candidate_ids']
    assert r['candidates'][0]['target_xyz'] is None
    assert r['candidates'][0]['localization']['reason']=='reused_rgb_geometry_mismatch'


def test_backprojection_respects_rotated_camera_axes():
    c=camera();c.c2w[:3,:3]=np.array([[0,0,1],[0,1,0],[-1,0,0]])
    mask=np.zeros((12,16),bool);mask[3:9,4:12]=True
    r,points,_=backproject_mask(mask,np.full(mask.shape,2.),np.ones(mask.shape),c,dict(min_points=3))
    np.testing.assert_allclose(points[:,0],5.)
    np.testing.assert_allclose(r['target_xyz'],[5,4,5])


def test_semantic_acceptance_and_disabled_localization_are_separate(tmp_path,monkeypatch):
    config,source=reuse_fixture(tmp_path,monkeypatch,True)
    config['localization']['mode']='disabled'
    _,r=run_offline(config,source)
    assert r['target_found'] and r['localization_status']=='NOT_ENABLED'
    assert r['sam2_calls']==0 and r['render_count']==0 and not r['localized_candidate_ids']


def test_vlm_review_budget_keeps_unreviewed_proposals_unconfirmed(tmp_path,monkeypatch):
    config,source=reuse_fixture(tmp_path,monkeypatch,True)
    config['offline']['max_vlm_candidates']=0
    _,r=run_offline(config,source)
    assert r['status']=='INCOMPLETE' and r['error'] is None
    assert not r['processing_complete'] and r['vlm_calls']==128
    assert r['candidates'][0]['status']=='review_skipped_budget'
