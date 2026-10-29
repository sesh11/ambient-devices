"""Source boundary checks for the standalone connector."""

import ast
from pathlib import Path

import tomllib


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "ambient_reachy"


def test_production_sources_have_no_hub_garage_or_openai_imports() -> None:
    forbidden = ("ambient_home", "garage", "garage_sdk", "openai")
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
