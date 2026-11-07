"""Single-image local VLM grounding + independent task-blind crop review.

This diagnostic does not localize in 3-D or claim multiview identity evidence.
"""
import argparse
from datetime import datetime
import html
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
import uuid
import numpy as np
from PIL import Image,ImageDraw
from gs_search.artifacts import write_json
from gs_search.cli import load_config
from gs_search.gaussian_projection import sha256
from gs_search.offline import rgb_panel,review_detection
from gs_search.perception import QwenVerifier
from gs_search.vlm_grounding import ground_rgb


def main():
    p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--target',default='绿色盘子 (green plate)')
    p.add_argument('--category',default='plate');p.add_argument('--config',default='configs/offline.json')
    p.add_argument('--device',default=None);args=p.parse_args()
    config=load_config(args.config);config['vlm'].update(category_prompt=args.category,max_new_tokens=512)
    if args.device: config['vlm']['device']=args.device
    root=Path(config['output_root'])/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_vlm_grounding_'+uuid.uuid4().hex[:6])
    root.mkdir(parents=True);(root/'crops').mkdir();(root/'reviews').mkdir()
    print(f'RESULT_DIR={root}',flush=True)
    started=time.perf_counter();source=Path(args.image).resolve();verifier=None
    report=dict(mode='single_image_direct_VLM_grounding',source_image=str(source),target=args.target,
                scene_geometry_used=False,confirmation='same-image model review only',candidates=[])
    write_json(root/'config.json',dict(vlm=config['vlm'],arguments=vars(args)))
    try:
        with Image.open(source) as im:
            if im.width*im.height>4_000_000: raise MemoryError('Diagnostic input exceeds four million pixels')
            rgb=np.asarray(im.convert('RGB'))
        report.update(source_sha256=sha256(source),width=rgb.shape[1],height=rgb.shape[0])
        Image.fromarray(rgb).save(root/'source.png')
        verifier=QwenVerifier(config['vlm'])
        print('[vlm] direct RGB grounding',flush=True)
        grounding=ground_rgb(rgb,args.target,verifier)
        report['grounding']=grounding;write_json(root/'grounding.json',grounding)
        for c in grounding['candidates']:
            print(f"[vlm] task-blind crop review {c['id']} box={c['box']}",flush=True)
            detection=SimpleNamespace(box=c['box'])
            Image.fromarray(rgb).crop(tuple(c['box'])).save(root/'crops'/f"{c['id']}.png")
            Image.fromarray(rgb_panel(rgb,detection)).save(root/'reviews'/f"{c['id']}.png")
            c['semantic_review']=review_detection(rgb,detection,args.target,verifier)
            c['accepted']=c['semantic_review'].get('matches_target') is True
            report['candidates'].append(c);write_json(root/'report.json',report)
        annotated=Image.fromarray(rgb);draw=ImageDraw.Draw(annotated)
        for c in report['candidates']:
            color='#22c55e' if c['accepted'] else '#f59e0b'
            draw.rectangle(c['box'],outline=color,width=3)
            x,y=c['box'][:2];draw.text((x,max(0,y-14)),f"{c['id']} {'ACCEPTED' if c['accepted'] else 'UNCONFIRMED'}",fill=color)
        annotated.save(root/'annotated.png')
        report.update(status='RGB_VLM_ACCEPTED' if any(c['accepted'] for c in report['candidates']) else 'INCOMPLETE',
                      accepted=sum(c['accepted'] for c in report['candidates']),error=None)
    except Exception as exc:
        report.update(status='ERROR',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        print(report['error']['traceback'],flush=True)
    finally:
        report.update(vlm_calls=verifier.calls if verifier else 0,model_info=getattr(verifier,'model_info',None),
                      inference_metrics=getattr(verifier,'metrics',[]),elapsed_seconds=time.perf_counter()-started)
        write_json(root/'report.json',report)
        cards=''.join(f'<h2>{c["id"]}</h2><img src="reviews/{c["id"]}.png">' for c in report['candidates'])
        (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px sans-serif;margin:24px}img{max-width:100%}pre{white-space:pre-wrap}</style>'
            '<h1>单图 VLM 直接定位</h1><p>绿色框：同图裁剪复核接受；橙色框：未确认。不是多视图确认或三维定位。</p>'
            '<img src="annotated.png">'+cards+'<pre>'+html.escape(json.dumps(report,ensure_ascii=False,indent=2))+'</pre>')
        print(f'REPORT={root}/index.html',flush=True)
        print(json.dumps({k:report.get(k) for k in ['status','accepted','vlm_calls','elapsed_seconds']},ensure_ascii=False),flush=True)
    return 1 if report.get('error') else 0


if __name__=='__main__': raise SystemExit(main())
