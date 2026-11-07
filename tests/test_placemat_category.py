"""Text alias and semantic gate regression; no real model inference here."""
import json
from gs_search.perception import category_identity,QwenVerifier


def test_placemat_spellings_normalize_without_equating_other_objects():
    for text in ['placemat','place mat','green place mat','place-mat','placemats']:
        assert category_identity(text,'placemat')=='placemat'
    for text in ['plate','tray','cup','table']:
        assert category_identity(text,'placemat')!='placemat'


def test_task_blind_place_mat_reaches_attribute_review_without_auto_accepting():
    v=QwenVerifier({'mode':'disabled','category_prompt':'placemat'});v.mode='TEST_DOUBLE'
    responses=iter([json.dumps(dict(views=[dict(id=1,object_type='place mat',identifiable=True)])),
                    json.dumps(dict(decision='reject',category_match=True,attributes_match=False,
                                    all_views_support=False,reason='TEST_DOUBLE wrong color'))])
    v._generate=lambda *_:next(responses)
    result=v.verify([None],'绿色餐垫')
    assert result['decision']=='reject' and result['matches_target'] is False
    assert result['reason']=='TEST_DOUBLE wrong color'
