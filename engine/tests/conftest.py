import pytest


@pytest.fixture(autouse=True)
def temper_home(tmp_path_factory, monkeypatch):
    """Settings and runs go to a scratch folder, never the developer's own."""
    monkeypatch.setenv("TEMPER_HOME", str(tmp_path_factory.mktemp("home")))
