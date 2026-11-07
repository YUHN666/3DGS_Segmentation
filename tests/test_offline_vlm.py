"""VLM backend integration uses explicitly labelled model doubles, not accuracy claims."""
import json
import numpy as np
import pytest
from gs_search.offline import run_offline
from gs_search.perception import QwenVerifier
from test_offline import reuse_fixture


@pytest.mark.parametrize('case',['accept','empty','invalid','budget'])
def test_vlm_shares_model_and_keeps_semantic_gate(tmp_path,monkeypatch,case):
    config,source=reuse_fixture(tmp_path,monkeypatch,case=='accept')
    config['vlm'].update(mode='local',review_input='focus_crop')
    config['target']={'description':'绿色餐垫','category':'placemat'}
    config['grounding']={'max_calls':2 if case=='budget' else 128}
    created=[]
    class VLMDouble:
        calls=0;metrics=[];groundings=0
        _parse=staticmethod(QwenVerifier._parse)
        def __init__(self,options): self.config=options;created.append(self)
        def _generate(self,images,prompt):
            assert len(images)==1 and images[0].dtype==np.uint8
            assert '绿色餐垫' in prompt
            self.calls+=1;self.groundings+=1
            if case=='invalid' and self.groundings==1: return 'TEST_DOUBLE malformed output'
            objects=[dict(label='TEST_DOUBLE green placemat',bbox_2d=[250,250,750,750])] if case=='accept' and self.groundings==1 else []
            return json.dumps(dict(objects=objects))
        def verify(self,images,target):
            assert target=='绿色餐垫' and images[0].dtype==np.uint8
            self.calls+=1
            return dict(matches_target=case=='accept',decision='accept' if case=='accept' else 'reject')
    monkeypatch.setattr('gs_search.perception.QwenVerifier',VLMDouble)
    root,r=run_offline(config,source)  # New target boxes always come from the current RGB request
    assert len(created)==1 and r['proposal_backend']=='vlm_grounding'
    assert r['vlm_grounding_calls']==(2 if case=='budget' else 128)
    assert r['reused_rgb_observations']==128 and r['error'] is None
    assert len(list((root/'grounding').glob('*.json')))==r['grounding_observations']
    if case=='accept':
        assert r['target_found'] and r['sam2_calls']==1
        assert r['vlm_calls']==129 and r['vlm_review_calls']==1
        assert r['candidates'][0]['score'] is None
        assert r['candidates'][0]['review_input']['format']=='unmarked_focus_crop'
        assert len(r['localized_candidate_ids'])==1 and r['processing_complete']
        assert '无数值置信度' in (root/'index.html').read_text()
    else:
        assert not r['target_found'] and r['sam2_calls']==0 and r['render_count']==0
        assert r['processing_complete']==(case=='empty')
        assert r['grounding_status']==('COMPLETE' if case=='empty' else 'PARTIAL')


def test_same_target_grounding_cache_reparsed_and_only_invalid_frame_retried(tmp_path,monkeypatch):
    config,source=reuse_fixture(tmp_path,monkeypatch,False)
    config['vlm']['mode']='local'
    config['grounding']={'max_calls':4,'invalid_retries':1}
    (source/'config.json').write_text(json.dumps(config));(source/'grounding').mkdir()
    for i in range(128):
        raw='[{"bbox_2d":' if i==0 else '[]'
        (source/'grounding'/f'{i:04d}.json').write_text(json.dumps(dict(raw_response=raw,prompt='TEST_DOUBLE cached prompt')))
    class VLM:
        calls=0;metrics=[]
        def __init__(self,*_): pass
        def _generate(self,images,prompt): self.calls+=1;return '{"objects":[]}'
    monkeypatch.setattr('gs_search.perception.QwenVerifier',VLM)
    root,r=run_offline(config,source,reuse_grounding=True)
    assert r['error'] is None and r['processing_complete'] and not r['target_found']
    assert r['grounding_cache_reads']==128 and r['reused_grounding_observations']==127
    assert r['vlm_grounding_calls']==r['vlm_calls']==1 and r['grounding_observations']==128
    assert (root/'grounding/0000_invalid_0.json').is_file()
    assert json.loads((source/'grounding/0000.json').read_text())['raw_response']=='[{"bbox_2d":'


def test_grounding_cache_cannot_reuse_different_target(tmp_path,monkeypatch):
    config,source=reuse_fixture(tmp_path,monkeypatch,False)
    config['vlm']['mode']='local'
    config['target']['description']='TEST_DOUBLE different target'
    _,r=run_offline(config,source,reuse_grounding=True)
    assert r['error']['type']=='ValueError' and 'target/category differs' in r['error']['message']
    assert r['vlm_calls']==0
