from pathlib import Path

from config import resolve_artifact_path


def test_resolve_artifact_path_uses_environment_override(tmp_path, monkeypatch):
    artifact_path = tmp_path / "custom-model.pkl"
    artifact_path.write_bytes(b"model")
    monkeypatch.setenv("EMS_MODEL_PATH", str(artifact_path))

    assert resolve_artifact_path("EMS_MODEL_PATH", "model.pkl") == artifact_path


def test_resolve_artifact_path_falls_back_to_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifact_path = workspace / "model.pkl"
    artifact_path.write_bytes(b"model")
    monkeypatch.chdir(workspace)

    assert resolve_artifact_path("EMS_MODEL_PATH", "model.pkl") == artifact_path
