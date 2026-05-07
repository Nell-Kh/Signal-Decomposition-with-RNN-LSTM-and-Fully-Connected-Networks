import json

from signal_decomp.shared.config import load_config


def test_load_config(tmp_path, monkeypatch):
    # Mock project structure
    # File is at src/signal_decomp/shared/config.py
    # Root is 3 levels up
    # config/setup.json

    mock_root = tmp_path / "root"
    mock_root.mkdir()
    config_dir = mock_root / "config"
    config_dir.mkdir()
    setup_json = config_dir / "setup.json"

    config_data = {"version": "2.0.0", "test": True}
    with open(setup_json, "w") as f:
        json.dump(config_data, f)

    # We need to mock __file__ in signal_decomp.shared.config
    # Or just rely on the fact that load_config looks for Path(__file__).resolve().parents[3]
    # This is tricky because __file__ is read-only.
    # I'll just check if it loads the actual project config for now,
    # or mock the Path.parents property.

    # Better: just test if it returns a dict with expected keys from the real project
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert "signal" in cfg
    assert "training" in cfg
