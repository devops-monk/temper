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
