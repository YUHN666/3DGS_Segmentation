"""Grounding parser/coordinates with a labelled model double, not real inference."""
import inspect
import json
import numpy as np
import pytest
from gs_search.perception import QwenVerifier
from gs_search.vlm_grounding import ground_rgb,parse_grounding


class VerifierDouble:
    _parse=staticmethod(QwenVerifier._parse)
    def __init__(self,response): self.response=response;self.inputs=[]
    def _generate(self,images,prompt):
        assert len(images)==1 and images[0].dtype==np.uint8
        self.inputs.append((images,prompt));return self.response


def test_grounding_only_receives_rgb_text_and_maps_normalized_boxes():
    v=VerifierDouble(json.dumps(dict(objects=[dict(label='green plate',bbox_2d=[100,200,300,250])])))
    rgb=np.zeros((600,800,3),np.uint8);r=ground_rgb(rgb,'绿色盘子',v)
    assert r['status']=='CANDIDATES' and len(v.inputs)==1
    np.testing.assert_allclose(r['candidates'][0]['box'],[80,120,240,150])
    assert r['candidates'][0]['score'] is None
    assert list(inspect.signature(ground_rgb).parameters)==['rgb','target_text','verifier','max_candidates']


def test_empty_or_invalid_response_never_fabricates_detection():
    rgb=np.zeros((600,800,3),np.uint8)
    for raw in ['{"objects":[]}','not JSON','{"objects":"plate"}']:
        assert ground_rgb(rgb,'green plate',VerifierDouble(raw))['candidates']==[]
    items=[dict(bbox_2d=b) for b in [[200,100,100,200],[-1,0,100,100],[0,0,1001,100],[0,0,1],
                                   [0,0,1,1],['bad',0,100,100]]]
    r=ground_rgb(rgb,'green plate',VerifierDouble(json.dumps(dict(objects=items))))
    assert not r['candidates'] and len(r['invalid'])==len(items)


def test_grounding_candidate_and_input_budgets():
    rgb=np.zeros((600,800,3),np.uint8)
    raw=json.dumps(dict(objects=[dict(bbox_2d=[100,100,200,200])]*3))
    r=ground_rgb(rgb,'plate',VerifierDouble(raw),max_candidates=1)
    assert len(r['candidates'])==1 and len(r['invalid'])==2
    with pytest.raises(TypeError): ground_rgb(rgb.astype(float),'plate',VerifierDouble(raw))
    with pytest.raises(ValueError): ground_rgb(rgb,'plate',VerifierDouble(raw),0)


def test_complete_json_arrays_supported_but_truncation_and_duplicate_keys_rejected():
    r=parse_grounding('```json\n[{"label":"mat","bbox_2d":[100,200,300,250],"confidence":0.98}]\n```',800,600)
    assert r['status']=='CANDIDATES' and r['candidates'][0]['score'] is None
    for text in ['[{"bbox_2d":[100,200,300,250]}',
                 '{"objects":[{"bbox_2d":[100,200,300,250],"bbox_2d":[100,200,300,250]}]}']:
        assert parse_grounding(text,800,600)['status']=='INVALID_RESPONSE'
