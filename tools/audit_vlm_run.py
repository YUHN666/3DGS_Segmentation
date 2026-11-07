"""Audit real VLM-only offline execution and cached evidence provenance."""
import argparse
import json
from pathlib import Path
import numpy as np
from gs_search.artifacts import write_json
from gs_search.gaussian_projection import sha256


def main():
    Path('artifacts').mkdir(exist_ok=True)
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('source');args=p.parse_args()
    root=Path(args.run).resolve();source=Path(args.source).resolve()
    report=json.loads((root/'report.json').read_text());rows=json.loads((root/'grounding.json').read_text())
    assert report['proposal_backend']=='vlm_grounding'
    assert report['vlm_calls']==report['vlm_grounding_calls']+report['vlm_review_calls']
    assert report['grounding_observations']==len(rows)==128
    assert report['vlm_grounding_calls']+report.get('reused_grounding_observations',0)>=128
    assert report['grounding_status']=='COMPLETE' and report['error'] is None
    for i in range(128):
        for directory,suffix in [('rgb','png'),('geometry','npz')]:
            assert sha256(root/directory/f'{i:04d}.{suffix}')==sha256(source/directory/f'{i:04d}.{suffix}')
    assert report['render_count']==0
    for candidate in report['candidates']:
        assert candidate['score'] is None and candidate['detection_provenance']['stage']=='vlm_grounding'
        raw=json.loads((root/candidate['detection_provenance']['response']).read_text())
        assert any(np.allclose(candidate['box'],proposal['box']) for proposal in raw['candidates'])
        assert candidate['semantic_review'] and candidate['review_input']['format']=='unmarked_focus_crop'
        if candidate['status']=='accepted_single_view_vlm':
            assert candidate['semantic_review']['matches_target'] is True
        else: assert candidate['localization']['status']=='NOT_ATTEMPTED'
    accepted=report['accepted_candidate_ids'];assert bool(accepted)==report['target_found']
    result=dict(status='PASS',run=str(root),proposal_backend='vlm_grounding',
        unchanged_rgb_and_geometry_files=256,render_count=0,vlm_grounding_calls=report['vlm_grounding_calls'],
        vlm_review_calls=report['vlm_review_calls'],vlm_calls=report['vlm_calls'],
        candidates=len(report['candidates']),vlm_accepted=len(accepted),localized=len(report['localized_candidate_ids']),
        score_is_null_not_fabricated=True,grounding_to_original_pixel_boxes_consistent=True,
        timing_seconds=report['timing_seconds'],projection=report.get('gaussian_projection',{}).get('directory'),
        limitation='Model evidence and numerical/provenance audit, not human ground truth or instance accuracy')
    write_json(root/'vlm_audit.json',result);write_json(Path('artifacts/offline_vlm_audit.json'),result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
