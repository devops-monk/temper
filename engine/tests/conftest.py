import pytest


@pytest.fixture(autouse=True)
def temper_home(tmp_path_factory, monkeypatch):
    """Settings and runs go to a scratch folder, never the developer's own."""
    monkeypatch.setenv("TEMPER_HOME", str(tmp_path_factory.mktemp("home")))
    from temper_engine import api_keys

    monkeypatch.setattr(api_keys, "_keys", None)  # read again from this test's home
    api_keys.log.clear()
    from temper_engine import usage

    monkeypatch.setattr(usage, "_days", None)
