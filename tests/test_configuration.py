import json
from pathlib import Path
from gs_search.configuration import current_config,target_identity
from gs_search.cli import parser
from gs_search.cli import load_config


def test_archived_category_is_read_without_mutating_saved_config():
    old={'target':{'description':'电脑','detector_prompt':'computer'},'detector':{'unused':True},
         'association':{'candidate_threshold':.75}}
    migrated=current_config(old)
    assert target_identity(old)==target_identity(migrated)==('电脑','computer')
    assert migrated['target']['category']=='computer' and 'detector' not in migrated
    assert 'candidate_threshold' not in migrated['association']
    assert old['target']['detector_prompt']=='computer' and 'detector' in old


def test_all_distributed_scene_configs_are_vlm_only():
    root=Path(__file__).resolve().parents[1]
    for path in (root/'configs').glob('*.json'):
        config=json.loads(path.read_text())
        if 'scene' not in config: continue
        assert 'detector' not in config,path
        assert config['target']['category'] and 'detector_prompt' not in config['target'],path
        assert 'candidate_threshold' not in config.get('association',{}),path
        assert config['vlm']['mode'] in ['local','service'],path
    args=parser().parse_args(['offline','--category','computer'])
    assert args.category=='computer'


def test_nested_local_config_paths_and_machine_overrides(tmp_path,monkeypatch):
    # Simulate a separate clone, with local settings excluded from publication.
    root=tmp_path/'clone'
    local=root/'configs/local';local.mkdir(parents=True)
    config={'scene':{'path':'scenes/example.ply'},'vlm':{'model_path':'models/qwen'},
            'segmentation':{'checkpoint':'models/sam.pt'},'output_root':'runs'}
    path=local/'machine.json';path.write_text(json.dumps(config))
    monkeypatch.setenv('GS_SEARCH_VLM_DEVICE','cuda:1')
    monkeypatch.setenv('GS_SEARCH_SAM2_CHECKPOINT',str(tmp_path/'external.pt'))
    loaded=load_config(path)
    assert loaded['scene']['path']==str(root/'scenes/example.ply')
    assert loaded['vlm']['model_path']==str(root/'models/qwen')
    assert loaded['output_root']==str(root/'runs')
    assert loaded['vlm']['device']=='cuda:1'
    assert loaded['segmentation']['checkpoint']==str(tmp_path/'external.pt')
    # An external prepared config retains its own relative path base.
    external=tmp_path/'external.json';external.write_text(json.dumps(config))
    assert load_config(external)['scene']['path']==str(tmp_path/'scenes/example.ply')
