import yaml
from types import SimpleNamespace

def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(
            **{k: _to_namespace(v) for k, v in obj.items()}
        )
    elif isinstance(obj, list):
        return [_to_namespace(x) for x in obj]
    return obj

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return _to_namespace(yaml.safe_load(f))
    
if __name__ == '__main__':
    cfg = load_config()
    print(cfg.train.epochs)