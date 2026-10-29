"""Source boundary checks for the standalone connector."""

import ast
import sys
import json
import hashlib
import subprocess
import importlib.util
from pathlib import Path

import tomllib


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "ambient_reachy"
VENDOR = SOURCE / "_vendor" / "pollen"


def test_production_sources_have_no_hub_garage_or_openai_imports() -> None:
    forbidden = ("ambient_home", "garage", "openai", "mcp", "reachy_mini_conversation_app")
    for path in SOURCE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
                if (node.module or "").startswith("ambient_device_sdk."):
                    raise AssertionError(f"private SDK import in {path}: {node.module}")
            else:
                continue
            for name in names:
                if any(name == prefix or name.startswith(f"{prefix}.") for prefix in forbidden):
                    raise AssertionError(f"forbidden import in {path}: {name}")


def test_sdk_is_not_vendored_and_openai_is_not_a_dependency() -> None:
    assert not (SOURCE / "ambient_device_sdk").exists()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert not any("openai" in dependency.lower() for dependency in project["project"]["dependencies"])
    assert not any(
        "reachy-mini-conversation-app" in dependency.lower() for dependency in project["project"]["dependencies"]
    )


def test_pollen_vendor_provenance() -> None:
    manifest = json.loads((VENDOR / "manifest.json").read_text(encoding="utf-8"))
    rewrites = (
        (
            b"ambient_reachy._vendor.pollen.tools._base",
            b"reachy_mini_conversation_app.tools.core_tools",
        ),
        (
            b"ambient_reachy._vendor.pollen.dance_emotion_moves",
            b"reachy_mini_conversation_app.dance_emotion_moves",
        ),
    )
    for upstream_path, entry in manifest.items():
        if entry.get("derived"):
            continue
        contents = (VENDOR / entry["vendored"]).read_bytes()
        for vendored_import, upstream_import in rewrites:
            contents = contents.replace(vendored_import, upstream_import)
        assert hashlib.sha256(contents).hexdigest() == entry["sha256"], upstream_path

    license_text = (VENDOR / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in license_text
    assert "Version 2.0" in license_text


def test_conversation_app_distribution_is_not_installed() -> None:
    assert importlib.util.find_spec("reachy_mini_conversation_app") is None


def test_vendor_imports_do_not_load_provider_or_mcp_modules() -> None:
    modules = (
        "ambient_reachy._vendor.pollen.moves",
        "ambient_reachy._vendor.pollen.tools.dance",
        "ambient_reachy._vendor.pollen.tools.stop_dance",
        "ambient_reachy._vendor.pollen.tools.play_emotion",
        "ambient_reachy._vendor.pollen.tools.stop_emotion",
        "ambient_reachy._vendor.pollen.tools.move_head",
        "ambient_reachy._vendor.pollen.tools.sweep_look",
        "ambient_reachy._vendor.pollen.tools.head_tracking",
    )
    script = (
        "import importlib, sys\n"
        f"for module in {modules!r}: importlib.import_module(module)\n"
        "assert 'openai' not in sys.modules\n"
        "assert 'mcp' not in sys.modules\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
