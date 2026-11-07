"""Semantic boundary: models accept RGB + text only, never geometry/depth/path.

Scores below are uncalibrated model scores. English category prompting is
independent of the optional VLM. Chinese/relational descriptions are preserved
and require VLM verification rather than silently translated or ignored.
"""
from dataclasses import dataclass, asdict
import numpy as np
from PIL import Image
import re


# Text-only category aliases, never scene labels or object coordinates. A VLM
# must not identify a cabinet and later relabel it as a bin to agree with a goal.
CATEGORY_ALIASES={
    'trash can':['trash can','trashcan','trash bin','garbage can','garbage bin','waste bin','rubbish bin','dustbin',
                 'wastebasket','waste basket','recycling bin','waste container'],
    'cabinet':['cabinet','cupboard','dresser','storage cabinet','sideboard'],
    'refrigerator':['refrigerator','fridge','freezer','mini fridge'],
    'chair':['chair','armchair','dining chair','office chair'],
    'table':['table','desk','counter','countertop'],
    'sofa':['sofa','couch'],
    'plant':['plant','potted plant','plant pot','planter'],
    'sink':['sink','washbasin'],
    'door':['door'],
    'lamp':['lamp','light fixture','ceiling light'],
    'placemat':['placemat','place mat'],
    'computer':['computer','desktop pc','pc','laptop','notebook computer'],
}


def category_identity(text,requested=None):
    words=re.sub(r'[^a-z0-9 ]',' ',str(text).lower())
    words=' '.join(words.split())
    if requested and words==requested: return requested
    hits={category for category,aliases in CATEGORY_ALIASES.items()
          if any(re.search(r'\b'+re.escape(alias)+r's?\b',words) for alias in aliases)}
    if len(hits)==1: return hits.pop()
    if not hits and requested and re.search(r'\b'+re.escape(requested)+r's?\b',words): return requested
    return None


@dataclass
class Detection:
    box: list  # xyxy in original RGB pixels
    score: float | None  # Optional historical score; VLM proposals use None
    label: str
    candidate_id: str | None = None
    review_status: str = 'proposal'
    provenance: dict | None = None

    def as_dict(self):
        return asdict(self)



class QwenVerifier:
    """Optional local Qwen3-VL or explicitly configured OpenAI-compatible service.

    Never supplies a service default or sends images without user configuration.
    Model prose is not converted into a success flag unless exact JSON parses.
    """
    def __init__(self, config):
        self.config, self.calls = config, 0
        self.mode = config.get("mode", "disabled")
        self.metrics=[]
        self.model_info=dict(mode=self.mode,model_path=config.get('model_path'),
                             dtype=config.get('dtype','bfloat16'),quantization=config.get('quantization','none'),
                             device=config.get('device','cuda:1'))
        if self.mode == "local":
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText
            self.processor = AutoProcessor.from_pretrained(config["model_path"], local_files_only=True)
            dtype={'bfloat16':torch.bfloat16,'float16':torch.float16,'float32':torch.float32}.get(config.get('dtype','bfloat16'))
            if dtype is None: raise ValueError('VLM dtype must be bfloat16/float16/float32')
            quant=config.get('quantization','none')
            kwargs=dict(local_files_only=True,dtype=dtype,attn_implementation='sdpa',
                        device_map={'':config.get('device','cuda:1')},low_cpu_mem_usage=True)
            if quant=='nf4':
                from transformers import BitsAndBytesConfig
                kwargs['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                                                               bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
            elif quant!='none': raise ValueError('VLM quantization must be none/nf4')
            self.model=AutoModelForImageTextToText.from_pretrained(config['model_path'],**kwargs).eval()
            self.model_info.update(parameters=int(self.model.num_parameters()),
                                   parameter_memory_bytes=int(self.model.get_memory_footprint()))
        elif self.mode not in ["disabled", "service"]:
            raise ValueError("VLM mode must be disabled/local/service")

    def verify(self, images: list, target_text: str):
        if self.mode == "disabled":
            return dict(status="not_enabled", matches_target=None,
                        limitation="English category only; attributes and relations unverified")
        import json
        # First identify the object without revealing the desired category.
        # Binary target-conditioned questions previously encouraged agreement
        # with a detector's cabinet-as-bin proposal.
        focused=getattr(self,'config',{}).get('review_input')=='focus_crop'
        introduction=(f'There are {len(images)} actual RGB crops, each centered on a proposed object. '
                      'Identify the actual object at the CENTER of EACH crop, independently of any task. '
                      'Do not mistake neighboring objects for the central object. ') if focused else (
                      f'There are {len(images)} actual RGB view panels. Each panel has the full frame on the left '
                      'with a red candidate rectangle, and a crop of that same object on the right. '
                      'Identify the actual object inside the rectangle independently in EACH view. ')
        blind_question=(introduction+
                        'Use a concrete everyday object type, not just a color or geometric shape. '
                        'Do not infer a function from a white rectangular surface alone. '
                        'If the object boundary or distinguishing parts cannot be seen, identifiable must be false. '
                        'Do not guess the task or desired object. Return only JSON: '
                        '{"views":[{"id":1,"object_type":"...","identifiable":false,'
                        '"visible_parts":["..."],"uncertainty":"..."}]}. '
                        f'Include exactly {len(images)} entries, with ids 1 through {len(images)}.')
        blind_raw=self._generate(images,blind_question)
        blind=self._parse(blind_raw)
        views=blind.get('views',[])
        valid=(isinstance(views,list) and len(views)==len(images) and
               all(isinstance(v,dict) and v.get('id')==i+1 and isinstance(v.get('object_type'),str)
                   and v['object_type'].strip() for i,v in enumerate(views)))
        if not valid or not all(v.get('identifiable') is True for v in views):
            return dict(status='reviewed',decision='uncertain',matches_target=False,
                        reason='target_blind_object_identity_not_established',blind_analysis=blind,
                        blind_response=blind_raw)
        category_prompt=getattr(self,'config',{}).get('category_prompt','')
        requested=category_identity(category_prompt) or str(category_prompt).strip().lower()
        observed=[category_identity(v['object_type'],requested) for v in views]
        if not requested or any(kind is None for kind in observed):
            return dict(status='reviewed',decision='uncertain',matches_target=False,
                        reason='task_blind_category_could_not_be_normalized',blind_analysis=blind,
                        blind_response=blind_raw,requested_category=requested,observed_categories=observed)
        if any(kind!=requested for kind in observed):
            return dict(status='reviewed',decision='reject',matches_target=False,
                        reason='task_blind_category_contradicts_requested_category',blind_analysis=blind,
                        blind_response=blind_raw,requested_category=requested,observed_categories=observed)
        question=(f'Target description: {target_text}\n'
                  'Independently identified object types from the previous task-blind inspection: '
                  +json.dumps(blind,ensure_ascii=False)+
                  ('\nCheck the central object in every actual RGB crop. ' if focused else
                   '\nCheck the marked object in every actual RGB view. ')+ 'Category must match first, then all '
                  'requested attributes and relationships. Similar color, a flat top, rectangular shape, '
                  'or stability across views do not establish object category. '
                  'Name contradictory evidence explicitly; if category-specific parts are missing, choose uncertain. '
                  'Do not reinterpret a storage cabinet, refrigerator, or other object as the requested category '
                  'only because its color fits. Do not assert cross-view identity. '
                  'Return only JSON {"decision":"accept|reject|uncertain","category_match":false,'
                  '"attributes_match":false,"all_views_support":false,"reason":"..."}.')
        response=self._generate(images,question)
        parsed=self._parse(response)
        decision=parsed.get('decision','uncertain')
        match=(decision=='accept' and all(parsed.get(key) is True for key in
                                         ['category_match','attributes_match','all_views_support']))
        if decision not in ['accept','reject','uncertain'] or (decision=='accept' and not match):
            decision='uncertain'
        return dict(status='reviewed',decision=decision,matches_target=match,
                    reason=parsed.get('reason','invalid_or_incomplete_semantic_result'),
                    blind_analysis=blind,blind_response=blind_raw,raw_response=response)

    @staticmethod
    def _parse(response):
        import json
        try:
            parsed=json.loads(response[response.index('{'):response.rindex('}')+1])
            return parsed if isinstance(parsed,dict) else {}
        except (ValueError,TypeError): return {}

    def _generate(self, images, question):
        import json
        import time
        started=time.perf_counter()
        self.calls += 1
        if self.mode == "local":
            import torch
            prepared=[]
            pixel_limit=self.config.get('max_image_pixels',1_048_576)
            for rgb in images:
                im=Image.fromarray(rgb)
                if im.width*im.height>pixel_limit:
                    scale=(pixel_limit/(im.width*im.height))**.5
                    im=im.resize((max(1,int(im.width*scale)),max(1,int(im.height*scale))),Image.Resampling.LANCZOS)
                prepared.append(im)
            content = [{"type": "image", "image": im} for im in prepared]
            messages = [{"role": "user", "content": content + [{"type": "text", "text": question}]}]
            inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                         return_dict=True, return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                ids = self.model.generate(**inputs, max_new_tokens=self.config.get('max_new_tokens',384), do_sample=False)
            response = self.processor.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            self.metrics.append(dict(seconds=time.perf_counter()-started,images=len(images),
                                     input_tokens=int(inputs.input_ids.shape[1]),
                                     generated_tokens=int(ids.shape[1]-inputs.input_ids.shape[1]),
                                     cuda_allocated_bytes=int(torch.cuda.memory_allocated(self.model.device)),
                                     cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(self.model.device))))
        else:
            import base64, io, os, urllib.request
            content = [{"type": "text", "text": question}]
            for rgb in images:
                stream = io.BytesIO()
                Image.fromarray(rgb).save(stream, format="PNG")
                content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                                base64.b64encode(stream.getvalue()).decode()}})
            body = json.dumps(dict(model=self.config["model"], messages=[dict(role="user", content=content)],
                                   max_tokens=self.config.get('max_new_tokens',384), temperature=0)).encode()
            headers = {"Content-Type": "application/json"}
            if self.config.get("api_key_env"):
                headers["Authorization"] = "Bearer " + os.environ[self.config["api_key_env"]]
            req = urllib.request.Request(self.config["url"], data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=self.config.get("timeout", 60)) as r:
                response = json.load(r)["choices"][0]["message"]["content"]
        return response
