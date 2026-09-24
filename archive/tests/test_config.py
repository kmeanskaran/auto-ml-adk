"""config/config.yml names the Ollama provider and the other tunables."""

from team.config import data_root
from team.config import detect_model_name
from team.config import load_config
from team.config import PROJECT_ROOT
from team.prompts import load_prompt


def test_yaml_selects_ollama_and_the_model(monkeypatch) -> None:
  monkeypatch.delenv("ML_DATA_ROOT", raising=False)
  monkeypatch.delenv("ML_RUNTIME_ROOT", raising=False)
  cfg = load_config()
  assert cfg.provider.name == "ollama"
  assert cfg.provider.model == "gpt-oss:20b-cloud"
  assert cfg.litellm_model == "ollama_chat/gpt-oss:20b-cloud"
  assert cfg.session.backend == "sqlite"
  assert cfg.provider.api_base.startswith("http://")
  assert cfg.runtime.root == "data"
  assert data_root() == PROJECT_ROOT / "data"


def test_model_env_overrides_the_yaml(monkeypatch) -> None:
  monkeypatch.setenv("ML_RUNTIME_MODEL", "scripted")
  assert detect_model_name() == "scripted"


def test_each_role_has_a_prompt() -> None:
  assert "NEEDED" in load_prompt("team.product_manager", "system.md")
  assert "NEEDED" in load_prompt("team.data_engineer", "system.md")
  assert "NEEDED" in load_prompt("team.data_scientist", "features.md")
  assert "NEEDED" in load_prompt("team.ml_engineer", "training.md")
  assert "NEEDED" in load_prompt("team.researcher", "schema.md")
