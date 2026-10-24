from collections import namedtuple

import httpx
from fastapi.testclient import TestClient

from temper_engine import doctor
from temper_engine.app import create_app

client = TestClient(create_app(token="secret"))
AUTH = {"Authorization": "Bearer secret"}
Usage = namedtuple("Usage", "total used free")


def offline(monkeypatch):
    monkeypatch.setattr(doctor, "check_network", lambda: doctor._check("network", "Hugging Face", "ok", "skipped in tests"))


def test_endpoint_returns_every_check(monkeypatch):
    offline(monkeypatch)
    checks = client.get("/api/doctor", headers=AUTH).json()["checks"]
    ids = {c["id"] for c in checks}
    assert {"engine", "packages", "disk", "memory", "gpu", "training", "llamacpp", "port", "network"} <= ids
    assert all(c["status"] in {"ok", "warn", "fail", "info"} for c in checks)
    assert next(c for c in checks if c["id"] == "packages")["status"] == "ok"


def test_endpoint_requires_token():
    assert client.get("/api/doctor").status_code == 401


def test_missing_package_offers_reinstall(monkeypatch):
    real = doctor.md.version

    def version(name):
        if name == "psutil":
            raise doctor.md.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(doctor.md, "version", version)
    check = doctor.check_packages()
    assert check["status"] == "fail"
    assert "psutil is missing" in check["detail"]
    assert check["action"] == "reinstall_engine"


def test_low_disk_space(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor.psutil, "disk_usage", lambda _path: Usage(100 * doctor.GB, 98 * doctor.GB, 2 * doctor.GB))
    check = doctor._disk(tmp_path, "Disk space", "disk")
    assert check["status"] == "fail" and "2.0 GB free" in check["detail"]
    monkeypatch.setattr(doctor.psutil, "disk_usage", lambda _path: Usage(100 * doctor.GB, 90 * doctor.GB, 10 * doctor.GB))
    assert doctor._disk(tmp_path, "Disk space", "disk")["status"] == "warn"


def test_port_fallback(monkeypatch):
    monkeypatch.setenv("TEMPER_PORT", "8765")
    assert doctor.check_port()["status"] == "ok"
    monkeypatch.setenv("TEMPER_PORT", "51000")
    monkeypatch.setattr(doctor, "_port_owner", lambda _port: "nginx")
    check = doctor.check_port()
    assert check["status"] == "warn" and "nginx" in check["detail"] and "51000" in check["detail"]
    monkeypatch.setattr(doctor, "_port_owner", lambda _port: None)
    assert doctor.check_port()["status"] == "info"


def test_nvidia_card_without_driver(monkeypatch):
    monkeypatch.setattr(doctor.platform, "system", lambda: "Linux")
    monkeypatch.setattr(doctor.hardware, "nvidia_gpus", lambda: [])
    monkeypatch.setattr(doctor.hardware, "amd_gpus", lambda: [])
    monkeypatch.setattr(doctor, "_gpu_names", lambda: ["NVIDIA Corporation AD102 [GeForce RTX 4090]"])
    check = doctor.check_gpu()
    assert check["status"] == "fail" and "RTX 4090" in check["detail"]


def test_old_nvidia_driver(monkeypatch):
    gpu = {"name": "Tesla T4", "driver": "470.82.01"}
    monkeypatch.setattr(doctor.hardware, "nvidia_gpus", lambda: [gpu])
    monkeypatch.setattr(doctor, "_run", lambda *_a, **_k: None)
    check = doctor.check_gpu()
    assert check["status"] == "warn" and "525" in check["fix"]


def test_network_failure_mentions_proxy(monkeypatch):
    def fail(*_a, **_k):
        raise httpx.ConnectError("no route")

    monkeypatch.setattr(doctor.httpx, "get", fail)
    monkeypatch.setenv("HTTPS_PROXY", "http://user:secret@proxy.local:3128")
    check = doctor.check_network()
    assert check["status"] == "warn"
    assert "proxy.local" in check["detail"] and "secret" not in check["detail"]


def test_a_crashing_check_does_not_hide_the_others(monkeypatch):
    def boom():
        raise RuntimeError("kaboom")

    monkeypatch.setattr(doctor, "CHECKS", [doctor.check_engine, boom])
    checks = doctor.run()
    assert checks[0]["status"] == "ok"
    assert checks[1]["status"] == "warn" and "kaboom" in checks[1]["detail"]
