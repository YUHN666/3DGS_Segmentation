"""Current target configuration plus read-only compatibility for saved runs."""
import copy


def target_identity(config):
    target=config['target']
    # Archived runs used this name for the independent review category.
    category=target.get('category',target.get('detector_prompt',''))
    return target['description'],category


def current_config(config):
    config=copy.deepcopy(config)
    if 'target' in config:
        _,category=target_identity(config)
        config['target'].pop('detector_prompt',None)
        config['target']['category']=category
    config.pop('detector',None)
    config.get('association',{}).pop('candidate_threshold',None)
    return config
