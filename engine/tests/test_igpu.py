from temper_engine import igpu

GB = igpu.GB


def test_a_mac_gets_the_command_to_raise_its_gpu_limit(monkeypatch):
    monkeypatch.setattr(igpu.hardware, "_run", lambda cmd: "0\n")
    out = igpu._mac(16 * GB)
    assert out["limit"] == out["default"] == int(16 * GB * 0.67) and not out["custom"]
    assert out["suggested"] == 12 * GB and out["commands"]["raise"] == "sudo sysctl iogpu.wired_limit_mb=12288"
    monkeypatch.setattr(igpu.hardware, "_run", lambda cmd: "14336\n")
    out = igpu._mac(16 * GB)
    assert out["limit"] == 14 * GB and out["custom"]
    assert igpu._mac(128 * GB)["suggested"] == 112 * GB  # at most 16 GB kept back


def test_an_amd_apu_is_found_by_its_small_carve_out(tmp_path, monkeypatch):
    def card(name, vendor, vram, gtt):
        device = tmp_path / name / "device"
        device.mkdir(parents=True)
        (device / "vendor").write_text(vendor + "\n")
        (device / "mem_info_vram_total").write_text(str(vram))
        (device / "mem_info_gtt_total").write_text(str(gtt))

    monkeypatch.setattr(igpu, "DRM", tmp_path)
    monkeypatch.setattr(igpu, "TTM", tmp_path / "pages_limit")
    card("card0", "0x1002", 24 * GB, 16 * GB)  # a graphics card: its own memory
    assert igpu.amd_apu(32 * GB) is None
    card("card1", "0x1002", 512 * 1024**2, 16 * GB)  # an APU: half a gigabyte carved out
    apu = igpu.amd_apu(32 * GB)
    assert apu == {"card": "card1", "vram": 512 * 1024**2, "gtt": 16 * GB}
    out = igpu._amd(32 * GB, apu)
    assert out["limit"] == 16 * GB and out["suggested"] == int(32 * GB - 32 * GB // 5)
    pages = out["suggested"] // 4096
    assert out["commands"]["params"] == f"ttm.pages_limit={pages} ttm.page_pool_size={pages}"
    assert out["commands"]["params"] in out["commands"]["raise"]
