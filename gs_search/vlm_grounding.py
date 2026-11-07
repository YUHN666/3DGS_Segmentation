"""Direct RGB/text candidate grounding, with no scene geometry input."""
import json
import re
import numpy as np


class VLMGrounder:
    """RGB-only adapter used by moving-camera search; shares the review model."""
    backend='vlm_grounding'

    def __init__(self,verifier,max_candidates=12):
        self.verifier=verifier;self.max_candidates=max_candidates;self.calls=0;self.last_response=None

    def detect(self,rgb,target_text):
        from .perception import Detection
        self.calls+=1
        self.last_response=ground_rgb(rgb,target_text,self.verifier,self.max_candidates)
        if self.last_response['status']=='INVALID_RESPONSE' or self.last_response['invalid']:
            raise ValueError('Invalid grounding response; raw response is retained for diagnosis')
        return [Detection(c['box'],None,c['label'],provenance=dict(stage='vlm_grounding',
                bbox_normalized=c['bbox_normalized'])) for c in self.last_response['candidates']]


def parse_grounding(raw,width,height,max_candidates=12):
    """Accept complete JSON object or list; never salvage truncated boxes."""
    def unique_keys(pairs):
        result={}
        for key,value in pairs:
            if key in result: raise ValueError('Duplicate JSON keys')
            result[key]=value
        return result
    cleaned=re.sub(r'^```(?:json)?\s*|\s*```$', '', str(raw).strip(),flags=re.IGNORECASE)
    try: parsed=json.loads(cleaned,object_pairs_hook=unique_keys)
    except (ValueError,TypeError): parsed=None
    objects=parsed if isinstance(parsed,list) else parsed.get('objects') if isinstance(parsed,dict) else None
    candidates=[];invalid=[]
    if not isinstance(objects,list):
        return dict(status='INVALID_RESPONSE',raw_response=raw,candidates=[],invalid=['Expected complete JSON objects list'])
    for index,item in enumerate(objects):
        if index>=max_candidates: invalid.append(dict(index=index,reason='candidate_budget'));continue
        try:
            bbox=np.asarray(item['bbox_2d'],float)
            if bbox.shape!=(4,) or not np.isfinite(bbox).all() or (bbox<0).any() or (bbox>1000).any():
                raise ValueError('Expected finite xyxy in 0..1000')
            if (bbox[2:]<=bbox[:2]).any(): raise ValueError('Reversed/empty box')
            box=bbox*np.array([width,height,width,height])/1000
            if (box[2:]-box[:2]<2).any(): raise ValueError('Box smaller than 2 original pixels')
            candidates.append(dict(id=f'candidate_{index:04d}',label=str(item.get('label','')),
                bbox_normalized=bbox.tolist(),box=box.tolist(),source='direct_VLM_RGB_grounding',score=None))
        except (KeyError,TypeError,ValueError) as exc: invalid.append(dict(index=index,reason=str(exc)))
    return dict(status='CANDIDATES' if candidates else ('INVALID_RESPONSE' if objects and invalid else 'NO_CANDIDATES'),
                raw_response=raw,coordinate_convention='xyxy / 1000 * [width,height,width,height]',candidates=candidates,invalid=invalid)


def ground_rgb(rgb,target_text,verifier,max_candidates=12):
    if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:
        raise TypeError('Grounding requires HxWx3 uint8 RGB')
    if not 1<=max_candidates<=64: raise ValueError('Invalid grounding candidate budget')
    prompt=(f'Locate all visible objects matching this target description: {target_text}. '
            'Inspect the actual image, including small objects. Do not invent unseen objects. '
            'Return tight 2D bounding boxes around individual matching objects, not a whole table or room. '
            'Use [x_min, y_min, x_max, y_max] coordinates normalized to 0..1000 across the entire image. '
            'If no matching object is visible, return an empty objects list. '
            f'Return at most {max_candidates} objects. Return ONLY JSON in this format: '
            '{"objects":[{"label":"observed object description","bbox_2d":[0,0,1000,1000]}]}. '
            'Do not include confidence scores or duplicate objects. Close the JSON and stop.')
    raw=verifier._generate([rgb],prompt)
    return dict(prompt=prompt,**parse_grounding(raw,rgb.shape[1],rgb.shape[0],max_candidates))
