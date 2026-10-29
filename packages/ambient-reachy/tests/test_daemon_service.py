"""Daemon-only LaunchAgent installer tests."""

import plistlib
import subprocess
from pathlib import Path

import pytest

from ambient_reachy import daemon_service as service


def _paths(tmp_path: Path) -> service.ServicePaths:
    daemon = tmp_path / "bin" / "reachy-mini-daemon"
    uv_bin = tmp_path / "bin" / "uv"
    daemon.parent.mkdir(parents=True, exist_ok=True)
    daemon.write_text("#!/bin/sh\n", encoding="utf-8")
    uv_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    return service.resolve_paths(daemon, uv_bin=uv_bin, home=tmp_path)


def test_render_plist_uses_stable_script_and_legacy_label(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    payload = plistlib.loads(service.render_plist(paths).encode())

    assert payload["Label"] == "ai.ambient-home.daemon"
    assert payload["ProgramArguments"] == ["/bin/sh", str(paths.script_path)]
    assert payload["WorkingDirectory"] == str(tmp_path)
    assert payload["EnvironmentVariables"]["REACHY_DAEMON_BIN"] == str(paths.daemon_bin)
    assert payload["EnvironmentVariables"]["UV_BIN"] == str(paths.uv_bin)
    assert payload["StandardErrorPath"] == str(tmp_path / "Library/Logs/ambient-home/daemon.log")


def test_render_plist_escapes_xml_paths(tmp_path: Path) -> None:
    home = tmp_path / "R&D <home>"
    daemon = home / "reachy-mini-daemon"
    uv_bin = home / "uv"
    daemon.parent.mkdir(parents=True)
    daemon.write_text("", encoding="utf-8")
    uv_bin.write_text("", encoding="utf-8")

    payload = plistlib.loads(service.render_plist(service.resolve_paths(daemon, uv_bin, home)).encode())

    assert payload["EnvironmentVariables"]["HOME"] == str(home)


def test_resolve_paths_requires_daemon_and_explicit_uv(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="reachy-mini-daemon"):
        service.resolve_paths(tmp_path / "missing-daemon", home=tmp_path)
    daemon = tmp_path / "daemon"
    daemon.write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="uv not found"):
        service.resolve_paths(daemon, uv_bin=tmp_path / "missing-uv", home=tmp_path)


def test_install_and_uninstall_manage_only_daemon_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[str, ...], bool]] = []
    paths = _paths(tmp_path)

    def fake_launchctl(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        calls.append((args, check))
        return subprocess.CompletedProcess(["launchctl", *args], 0, "", "")

    monkeypatch.setattr(service, "launchctl", fake_launchctl)
    monkeypatch.setattr(service, "domain", lambda: "gui/501")

    installed = service.install(paths)

    assert installed == paths.plist_path
    assert paths.plist_path.is_file()
    assert paths.script_path.is_file()
    assert paths.script_path.stat().st_mode & 0o111
    assert paths.log_dir.is_dir()
    assert calls == [
        (("bootout", "gui/501/ai.ambient-home.daemon"), False),
        (("bootstrap", "gui/501", str(paths.plist_path)), False),
    ]

    service.uninstall(paths)

    assert not paths.plist_path.exists()
    assert not paths.script_path.exists()
    assert calls[-1] == (("bootout", "gui/501/ai.ambient-home.daemon"), False)


def test_install_dry_run_touches_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(service, "launchctl", lambda *args, **kwargs: pytest.fail("launchctl must not run"))

    service.install(paths, dry_run=True)

    assert not paths.agents_dir.exists()
    assert "ai.ambient-home.daemon" in capsys.readouterr().out


def test_cli_install_dry_run_accepts_explicit_paths(tmp_path: Path, capsys) -> None:
    paths = _paths(tmp_path)

    assert (
        service.install_main(["--dry-run", "--daemon-bin", str(paths.daemon_bin), "--uv-bin", str(paths.uv_bin)]) == 0
    )
    assert "ai.ambient-home.daemon" in capsys.readouterr().out
