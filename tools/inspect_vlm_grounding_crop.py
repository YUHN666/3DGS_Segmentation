"""RGB-only diagnostic: inspect context around model-predicted boxes, task blind."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from gs_search.artifacts import write_json
from gs_search.perception import QwenVerifier


def main():
    p=argparse.ArgumentParser();p.add_argument('run_dir');args=p.parse_args()
    root=Path(args.run_dir).resolve();report=json.loads((root/'report.json').read_text())
    config=json.loads((root/'config.json').read_text());started=time.perf_counter()
    rgb=np.asarray(Image.open(root/'source.png').convert('RGB'));h,w=rgb.shape[:2]
    verifier=QwenVerifier(config['vlm']);results=[]
    prompt=('Identify the object at the CENTER of this image crop independently of any task. '
            'Describe its actual everyday object category, color, shape, and distinguishing physical features. '
            'Do not mistake neighboring objects for the central object. '
            'If it could be multiple categories, list alternatives and explain uncertainty. '
            'Return only JSON: {"object_type":"...","color":"...","shape":"...",'
            '"visible_features":["..."],"alternative_categories":["..."],"uncertainty":"..."}.')
    for c in report['candidates']:
        box=np.asarray(c['box'],float);margin=max(16,.25*max(box[2:]-box[:2]))
        region=[max(0,int(np.floor(box[0]-margin))),max(0,int(np.floor(box[1]-margin))),
                min(w,int(np.ceil(box[2]+margin))),min(h,int(np.ceil(box[3]+margin)))]
        crop=Image.fromarray(rgb).crop(tuple(region));scale=768/max(crop.size)
        crop=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.LANCZOS)
        name=f'crop_inspection_{c["id"]}.png';crop.save(root/name)
        raw=verifier._generate([np.asarray(crop)],prompt)
        results.append(dict(candidate_id=c['id'],region_original_pixels=region,image=name,prompt=prompt,
                            raw_response=raw,parsed=verifier._parse(raw)))
    data=dict(status='DIAGNOSTIC_ONLY',scope='Task-blind RGB crop classification; no new search confirmation',
              region_policy='Model-predicted box plus max(16 pixels, 25% of largest side) context',
              target_text_provided_to_model=False,results=results,vlm_calls=verifier.calls,
              inference_metrics=verifier.metrics,elapsed_seconds=time.perf_counter()-started)
    write_json(root/'crop_inspection.json',data);report['additional_crop_inspection']=data
    report['total_vlm_calls']=report.get('total_vlm_calls',report['vlm_calls'])+verifier.calls
    write_json(root/'report.json',report)
    page=root/'index.html'
    import html
    page.write_text(page.read_text()+'<h2>无目标文本的近景检查</h2>'+''.join(
        f'<img src="{r["image"]}"><pre>'+html.escape(json.dumps(r['parsed'],ensure_ascii=False,indent=2))+'</pre>' for r in results))
    print(json.dumps(data,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
