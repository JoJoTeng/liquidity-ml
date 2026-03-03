"""Load project configuration from config/config.yaml."""

from pathlib import Path
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_config_cache = None


def load_config() -> dict:
    global _config_cache
    if _config_cache is None:
        with open(_PROJECT_ROOT / "config" / "config.yaml") as f:
            _config_cache = yaml.safe_load(f)
    return _config_cache


def get_data_dir() -> Path:
    d = _PROJECT_ROOT / load_config()["project"]["data_dir"]
    d.mkdir(exist_ok=True)
    return d


def get_output_dir() -> Path:
    d = _PROJECT_ROOT / load_config()["project"]["output_dir"]
    d.mkdir(exist_ok=True)
    return d
