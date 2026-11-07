"""Control/semantic-gate regressions; synthetic doubles are not model accuracy tests."""
import json
import numpy as np
from gs_search.memory import Evidence,Candidate,ObservationMemory,associate
from gs_search.perception import Detection,QwenVerifier
from test_memory import camera


def pair(clipped=False):
    rng=np.random.default_rng(17)
    xyz=rng.uniform([-.8,-.6,3],[.8,.6,4],(80,3))
    descriptors=rng.uniform(0,250,(80,128)).astype(np.float32)
    sightings=[]
    for i,x in enumerate([-.3,.3]):
        c=camera(x);q=xyz-c.position
        uv=q[:,:2]/q[:,2:]*400+[200,150]
        box=[10,10,390,300 if clipped else 290]
        sightings.append(Evidence(i,c,Detection(box,.9,'trash can'),
                                  rng.integers(0,256,(300,400,3),dtype=np.uint8),
                                  uv.astype(np.float32),descriptors,str(i)))
    link=associate(*sightings,{})
    assert link['accepted']
    return Candidate('fixture',sightings,[link])


def test_clipped_stable_cabinet_cannot_be_confirmed_even_with_yes_verifier():
    class AlwaysYes:
        mode='TEST_DOUBLE';calls=0
        def verify(self,*args): self.calls+=1;return dict(matches_target=True)
    m=ObservationMemory({'max_box_area_fraction':.98})
    c=pair(clipped=True);v=AlwaysYes()
    assert not m.confirm(c,v,'白色垃圾桶',True)
    assert v.calls==0 and c.status=='pending_wider_view'
    assert not c.confirmation_observation_ids


def test_explicit_semantic_rejection_is_terminal_and_not_requeried():
    class RejectCabinet:
        mode='TEST_DOUBLE';calls=0
        def verify(self,*args): self.calls+=1;return dict(matches_target=False,decision='reject',reason='cabinet')
    m=ObservationMemory({'max_box_area_fraction':.98})
    c=pair();v=RejectCabinet()
    assert not m.confirm(c,v,'白色垃圾桶',True)
    assert c.status=='rejected_semantic'
    assert not m.confirm(c,v,'白色垃圾桶',True) and v.calls==1
    assert len(c.review_history)==1


def test_target_blind_first_pass_and_strict_schema_fail_closed():
    v=QwenVerifier.__new__(QwenVerifier);v.mode='TEST_DOUBLE';v.calls=0;v.config={'category_prompt':'trash can'}
    questions=[]
    answers=[json.dumps({'views':[{'id':i,'object_type':'waste bin','identifiable':True} for i in [1,2]]}),
             '{"matches_target":true}']
    def generate(images,question):
        questions.append(question);v.calls+=1;return answers.pop(0)
    v._generate=generate
    result=v.verify([np.zeros((20,20,3),np.uint8)]*2,'白色垃圾桶')
    assert '白色垃圾桶' not in questions[0] and '白色垃圾桶' in questions[1]
    assert result['matches_target'] is False and result['decision']=='uncertain'


def test_cabinet_class_cannot_be_overridden_by_target_conditioned_yes():
    v=QwenVerifier.__new__(QwenVerifier);v.mode='TEST_DOUBLE';v.calls=0;v.config={'category_prompt':'trash can'}
    def generate(images,question):
        v.calls+=1
        assert v.calls==1,'Contradictory blind category must stop before a leading yes/no pass'
        return json.dumps({'views':[{'id':i,'object_type':'white storage cabinet','identifiable':True} for i in [1,2]]})
    v._generate=generate
    result=v.verify([np.zeros((20,20,3),np.uint8)]*2,'白色垃圾桶')
    assert result['matches_target'] is False and result['decision']=='reject'
    assert result['observed_categories']==['cabinet','cabinet'] and v.calls==1


def test_endless_new_false_candidates_force_translation_not_verify_loop(tmp_path):
    from test_controller import setup
    c=setup(tmp_path,False)
    c.config['search'].update(max_observations=12,max_path_length=12,max_seconds=90,max_verify_attempts=20,
                              max_consecutive_verify_observations=3,min_explore_progress=1.5)
    r=c.run()
    assert r['error'] is None and r['status']=='INCOMPLETE'
    events=c.artifacts.events
    assert any(e.get('reason')=='local_verification_budget_requires_spatial_exploration' for e in events)
    assert any(e['kind']=='exploration_commitment_completed' and e['distance']>=1.5-1e-6 for e in events)
    positions=np.array([e['position'] for e in c.env.trajectory])
    assert np.ptp(positions[:,[0,2]],axis=0).max()>1.5
    c.env.grid.validate_path(positions)



def test_explicit_fallback_names_actual_backend_and_validates_segments():
    from test_navigation import grid
    from gs_search.navigation import PlanningError
    from gs_search.planner_fallback import ExplicitPlannerFallback
    class RejectOptimizedPolyline:
        backend='TEST_PRIMARY';last_metadata={'attempts':[{'status':'rejected','error':'voxel collision'}]}
        def plan(self,*args): raise PlanningError('optimized curve intersects occupied voxel')
    g=grid()
    planner=ExplicitPlannerFallback(RejectOptimizedPolyline(),g)
    path=planner.plan([1.1,1.5,1.1],[4.9,1.5,1.1])
    assert g.validate_path(path)
    assert planner.last_metadata['backend']=='occupancy_astar_3d'
    assert planner.last_metadata['fallback_used'] is True
