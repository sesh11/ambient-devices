"""Install and remove the Reachy daemon LaunchAgent on macOS."""

import os
import sys
import shutil
import string
import argparse
import platform
import subprocess
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Sequence
from xml.sax.saxutils import escape
from importlib.resources import files


DAEMON_LABEL = "ai.ambient-home.daemon"
APP_LABEL = "ai.ambient-home.app"


@dataclass(frozen=True)
class ServicePaths:
    """Per-machine locations rendered into the daemon LaunchAgent."""

    daemon_bin: Path
    uv_bin: Path
    home: Path

    @property
    def log_dir(self) -> Path:
        """Return the shared ambient-home daemon log directory."""
        return self.home / "Library" / "Logs" / "ambient-home"

    @property
    def agents_dir(self) -> Path:
        """Return the current user's LaunchAgents directory."""
        return self.home / "Library" / "LaunchAgents"

    @property
    def script_path(self) -> Path:
        """Return the installed daemon runner path."""
        return self.home / "Library" / "Application Support" / "ambient-reachy" / "run-daemon.sh"

    @property
    def plist_path(self) -> Path:
        """Return the daemon LaunchAgent plist path."""
        return self.agents_dir / f"{DAEMON_LABEL}.plist"

    @property
    def path(self) -> str:
        """PATH exported to launchd."""
        candidates = [str(self.daemon_bin.parent)]
        if self.uv_bin.is_absolute():
            candidates.append(str(self.uv_bin.parent))
        candidates.extend(["/opt/homebrew/bin", "/usr/local/bin"])
        candidates.extend(["/usr/bin", "/bin", "/usr/sbin", "/sbin"])
        return ":".join(dict.fromkeys(candidates))

    def substitutions(self) -> dict[str, str]:
        """Return XML-safe values for the plist template."""
        return {
            "daemon_bin": str(self.daemon_bin),
            "uv_bin": str(self.uv_bin),
            "home": str(self.home),
            "log_dir": str(self.log_dir),
            "path": self.path,
            "script_path": str(self.script_path),
        }


def resolve_paths(
    daemon_bin: Path | None = None,
    uv_bin: Path | None = None,
    home: Path | None = None,
    *,
    require_daemon: bool = True,
) -> ServicePaths:
    """Resolve executable and LaunchAgent paths."""
    resolved_daemon = (
        (daemon_bin or Path(sys.executable).resolve().with_name("reachy-mini-daemon")).expanduser().resolve()
    )
    if require_daemon and not resolved_daemon.is_file():
        raise FileNotFoundError(f"reachy-mini-daemon not found at {resolved_daemon}; pass --daemon-bin")
    found_uv = shutil.which("uv")
    if uv_bin is not None:
        resolved_uv = uv_bin.expanduser().resolve()
        if not resolved_uv.is_file():
            raise FileNotFoundError(f"uv not found at {resolved_uv}")
    elif found_uv:
        resolved_uv = Path(found_uv).resolve()
    else:
        resolved_uv = Path("uv")
    return ServicePaths(resolved_daemon, resolved_uv, (home or Path.home()).expanduser().resolve())


def render_plist(paths: ServicePaths) -> str:
    """Render the packaged daemon LaunchAgent template."""
    template = files("ambient_reachy").joinpath("launchd", f"{DAEMON_LABEL}.plist").read_text(encoding="utf-8")
    return string.Template(template).substitute({key: escape(value) for key, value in paths.substitutions().items()})


def launchctl(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a launchctl subcommand."""
    return subprocess.run(["launchctl", *args], check=check, capture_output=True, text=True)


def domain() -> str:
    """Return the current user's launchd GUI domain."""
    return f"gui/{os.getuid()}"


def install(paths: ServicePaths, *, dry_run: bool = False) -> Path:
    """Install the daemon plist and stable packaged runner script."""
    if dry_run:
        print(f"# {paths.plist_path}\n{render_plist(paths)}")
        return paths.plist_path
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    paths.agents_dir.mkdir(parents=True, exist_ok=True)
    paths.script_path.parent.mkdir(parents=True, exist_ok=True)
    script = files("ambient_reachy").joinpath("scripts", "run-daemon.sh").read_text(encoding="utf-8")
    paths.script_path.write_text(script, encoding="utf-8")
    paths.script_path.chmod(0o755)
    paths.plist_path.write_text(render_plist(paths), encoding="utf-8")
    launchctl("bootout", f"{domain()}/{DAEMON_LABEL}", check=False)
    result = launchctl("bootstrap", domain(), str(paths.plist_path), check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"launchctl bootstrap {DAEMON_LABEL} failed: {result.stderr.strip() or result.stdout.strip()}"
        )
    print(f"installed {DAEMON_LABEL} -> {paths.plist_path}")
    print(f"logs: {paths.log_dir}/daemon.log")
    return paths.plist_path


def uninstall(paths: ServicePaths, *, dry_run: bool = False) -> None:
    """Boot out and remove the daemon agent and its stable runner script."""
    if dry_run:
        print(f"would bootout {domain()}/{DAEMON_LABEL} and remove {paths.plist_path}")
        return
    launchctl("bootout", f"{domain()}/{DAEMON_LABEL}", check=False)
    paths.plist_path.unlink(missing_ok=True)
    paths.script_path.unlink(missing_ok=True)
    print(f"removed {DAEMON_LABEL}")


def _parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--daemon-bin", type=Path, default=None, help="path to reachy-mini-daemon")
    parser.add_argument("--uv-bin", type=Path, default=None, help="path to uv (fallback only)")
    parser.add_argument("--dry-run", action="store_true", help="print the plist without touching files or launchd")
    return parser


def _require_macos(dry_run: bool) -> None:
    if platform.system() != "Darwin" and not dry_run:
        raise SystemExit("LaunchAgents are macOS-only; use --dry-run elsewhere to preview the plist")


def install_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ambient-reachy-install-daemon."""
    args = _parser("Install the Reachy Mini daemon LaunchAgent.").parse_args(argv)
    _require_macos(args.dry_run)
    try:
        install(resolve_paths(args.daemon_bin, args.uv_bin), dry_run=args.dry_run)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def uninstall_main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ambient-reachy-uninstall-daemon."""
    args = _parser("Remove the Reachy Mini daemon LaunchAgent.").parse_args(argv)
    _require_macos(args.dry_run)
    try:
        paths = resolve_paths(args.daemon_bin, args.uv_bin, require_daemon=False)
        uninstall(paths, dry_run=args.dry_run)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
