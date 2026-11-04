"""The numbers shown in the videos must match what the simulations produce today."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_exporter():
    spec = importlib.util.spec_from_file_location("export_data", ROOT / "videos" / "export_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_video_data_is_in_sync_with_the_simulations():
    text = (ROOT / "videos" / "src" / "data.js").read_text()
    committed = json.loads(text[text.index("{") : text.rindex("}") + 1])
    fresh = json.loads(json.dumps(_load_exporter().collect()))
    assert committed == fresh, "run `python videos/export_data.py` and re-render the videos"


def test_every_video_script_is_listed_in_the_player():
    html = (ROOT / "videos" / "index.html").read_text()
    for script in sorted((ROOT / "videos" / "src").glob("*.js")):
        assert f'src="src/{script.name}"' in html
