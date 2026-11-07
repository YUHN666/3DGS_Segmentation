"""Validate raw Gaussian preservation and actually render exported 3DGS PLYs.

Use run.sh's compiler/CUDA cache environment when invoking this GPU check.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from plyfile import PlyData
from gs_search.artifacts import write_json
from gs_search.camera import Camera
from gs_search.gaussian_projection import sha256
from gs_search.render import GSplatRenderer
from gs_search.scene import load_scene


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run_dir');args=parser.parse_args()
    root=Path(args.run_dir).resolve();report=json.loads((root/'report.json').read_text())
    config=json.loads((root/'config.json').read_text());projection=report['gaussian_projection']
    dest=root/projection['directory'];output=dest/'render_validation';output.mkdir(exist_ok=True)
    raw=PlyData.read(config['scene']['path'])['vertex'].data
    labels=PlyData.read(dest/'scene_target_labels.ply')['vertex'].data
    highlighted=PlyData.read(dest/'scene_target_highlighted.ply')['vertex'].data
    target=PlyData.read(dest/'target_gaussians.ply')['vertex'].data
    indices=np.load(dest/'target_gaussian_indices.npy');keep=labels['search_target'].astype(bool)
    np.testing.assert_array_equal(indices,np.flatnonzero(keep))
    assert len(raw)==len(labels)==len(highlighted) and len(target)==len(indices)>0
    for name in raw.dtype.names:
        np.testing.assert_array_equal(raw[name],labels[name])
        np.testing.assert_array_equal(raw[name][indices],target[name])
        if name.startswith(('f_dc_','f_rest_')):
            np.testing.assert_array_equal(raw[name][~keep],highlighted[name][~keep])
        else: np.testing.assert_array_equal(raw[name],highlighted[name])
    assert sha256(config['scene']['path'])==projection['source_sha256']
    cameras=json.loads((root/'cameras.json').read_text())['cameras']
    candidates={c['id']:c for c in report['candidates']}
    views=projection['views'];rendered={}
    for kind,path in [('original',Path(config['scene']['path'])),('highlighted',dest/'scene_target_highlighted.ply'),
                      ('target',dest/'target_gaussians.ply')]:
        scene_config={**config['scene'],'path':str(path)}
        renderer=GSplatRenderer(load_scene(scene_config),config['renderer'])
        for view in views:
            obs=view['observation_id'];rgb=renderer.render(Camera.from_dict(cameras[obs]))
            Image.fromarray(rgb).save(output/f'{obs:04d}_{kind}.png');rendered[kind,obs]=rgb
        del renderer
    results=[];cards=[];montage=Image.new('RGB',(1080,6*230),(25,30,40));draw=ImageDraw.Draw(montage)
    for index,view in enumerate(views):
        obs=view['observation_id'];candidate=candidates[view['candidate_id']]
        original=rendered['original',obs];high=rendered['highlighted',obs];subset=rendered['target',obs]
        saved=np.asarray(Image.open(root/'rgb'/f'{obs:04d}.png').convert('RGB'))
        error=int(np.abs(original.astype(int)-saved.astype(int)).max());assert error<=2
        mask=np.asarray(Image.open(root/candidate['localization']['mask']))>0
        changed=np.max(np.abs(high.astype(int)-original.astype(int)),axis=2)>20
        assert changed[mask].any() and original.std()>2 and subset.max()>20
        results.append(dict(observation_id=obs,candidate_id=candidate['id'],rgb_source_max_error=error,
            changed_pixels=int(changed.sum()),changed_inside_mask=int((changed&mask).sum()),
            accepted_mask_pixels=int(mask.sum()),original_rgb_std=float(original.std()),target_rgb_std=float(subset.std())))
        x0,y0,x1,y1=candidate['box'];padding=max(24,(x1-x0)*.4,(y1-y0)*.2)
        crop=(max(0,int(x0-padding)),max(0,int(y0-padding)),min(original.shape[1],int(x1+padding)),min(original.shape[0],int(y1+padding)))
        for col,kind in enumerate(['original','highlighted','target']):
            tile=Image.fromarray(rendered[kind,obs]).crop(crop);tile.thumbnail((350,202))
            montage.paste(tile,(col*360,index*230+25));draw.text((col*360+4,index*230+4),f'{obs:04d} {kind}',fill='white')
        cards.append(f'<h2>{candidate["id"]} · observation {obs}</h2>'+''.join(
            f'<a href="{obs:04d}_{kind}.png"><img width="32%" src="{obs:04d}_{kind}.png" title="{kind}"></a>'
            for kind in ['original','highlighted','target']))
    montage.save(output/'comparison.jpg',quality=95)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>真实 gsplat 渲染检查</h1>'
        '<p>每行：原场景 / 原场景高亮目标 / 原始目标高斯子集。点击查看全分辨率。</p>'
        '<img width="100%" src="comparison.jpg">'+''.join(cards))
    audit=dict(status='PASS',backend='actual gsplat RGB',source_ply_unchanged=True,
               all_original_fields_preserved_in_labels=True,geometry_opacity_rotation_unchanged_in_highlight=True,
               subset_rows_exactly_match_original=True,selected_gaussians=len(indices),full_scene_gaussians=len(raw),
               render_count=3*len(views),views=results,
               limitation='Pixel comparisons check export/render consistency, not manual semantic segmentation accuracy')
    write_json(output/'validation.json',audit);write_json(Path('artifacts/gaussian_projection_validation.json'),audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2));print(f'RENDER_CHECK={output}/index.html')


if __name__=='__main__': main()
