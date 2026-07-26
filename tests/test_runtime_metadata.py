from pathlib import Path

from src.common import ROOT, load_config, runtime_metadata


def test_runtime_metadata_uses_repository_relative_config_path() -> None:
    config = load_config(ROOT / "config" / "config.yaml")
    metadata = runtime_metadata(config)

    assert metadata["config_path"] == str(Path("config") / "config.yaml")
    assert not Path(metadata["config_path"]).is_absolute()
