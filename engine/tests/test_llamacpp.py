import pytest

from temper_engine import llamacpp


@pytest.mark.parametrize(
    ("system", "libc", "too_old"),
    [
        ("Linux", ("glibc", "2.35"), True),  # Ubuntu 22.04
        ("Linux", ("glibc", "2.36"), True),  # Debian 12
        ("Linux", ("glibc", "2.38"), False),
        ("Linux", ("glibc", "2.41"), False),
        ("Linux", ("", ""), False),  # musl or unknown: let the download say
        ("Darwin", ("", ""), False),
    ],
)
def test_glibc_too_old(monkeypatch, system, libc, too_old):
    monkeypatch.setattr(llamacpp.platform, "system", lambda: system)
    monkeypatch.setattr(llamacpp.platform, "libc_ver", lambda: libc)
    why = llamacpp.glibc_too_old()
    assert bool(why) is too_old
    if why:
        assert libc[1] in why and "PATH" in why


def test_install_refuses_before_downloading(monkeypatch):
    monkeypatch.setattr(llamacpp.platform, "system", lambda: "Linux")
    monkeypatch.setattr(llamacpp.platform, "libc_ver", lambda: ("glibc", "2.35"))
    monkeypatch.setattr(llamacpp, "_download", lambda *a, **k: pytest.fail("downloaded"))
    with pytest.raises(RuntimeError, match="glibc 2.38"):
        llamacpp.install()


def release(tag, days_old, names, digest=True):
    from datetime import datetime, timedelta, timezone

    published = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat().replace("+00:00", "Z")
    return {"tag_name": tag, "published_at": published, "draft": False, "prerelease": True,
            "assets": [{"name": n, "digest": f"sha256:{n[:8]}" if digest else None} for n in names]}


@pytest.fixture
def mac(monkeypatch):
    monkeypatch.setattr(llamacpp.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(llamacpp.platform, "machine", lambda: "arm64")
    llamacpp._newer = (0.0, None)
    return lambda tag: f"llama-{tag}-bin-macos-arm64.tar.gz"


def test_a_newer_build_is_a_week_old_and_checked(monkeypatch, mac):
    releases = [release("b99003", 1, [mac("b99003")]), release("b99002", 9, [mac("b99002")], digest=False), release("b99001", 10, [mac("b99001")])]
    monkeypatch.setattr(llamacpp, "_releases", lambda: iter(releases))
    found = llamacpp.newer(fresh=True)
    assert found["tag"] == "b99001" and found["archives"] == [(mac("b99001"), "llama-b9")]  # too new, then unchecked, are skipped


def test_update_keeps_the_build_before_for_rollback(monkeypatch, mac, tmp_path):
    import io
    import tarfile

    def archive(*a, **k):
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w:gz") as tar:
            info = tarfile.TarInfo("build/bin/llama-server" + (".exe" if llamacpp.os.name == "nt" else ""))
            info.size = 2
            tar.addfile(info, io.BytesIO(b"#!"))
        return data.getvalue()

    monkeypatch.setattr(llamacpp, "_download", archive)
    llamacpp.install()  # the pinned build
    assert llamacpp.active_tag() == llamacpp.TAG
    monkeypatch.setattr(llamacpp, "newer", lambda fresh=False: {"tag": "b99001", "published": "", "archives": [(mac("b99001"), "x")]})
    assert llamacpp.update() == "b99001"
    assert llamacpp.active_tag() == "b99001" and llamacpp.installed_build().name.startswith("b99001")
    assert llamacpp.binary("llama-server").startswith(str(llamacpp.root() / "b99001"))
    assert llamacpp.rollback() == llamacpp.TAG
    assert llamacpp.active_tag() == llamacpp.TAG and not (llamacpp.root() / "chosen.json").exists()
