"""Build and install ``aisdb`` from source on Pythons it ships no wheel for.

Why this exists
---------------
aisdb publishes Windows wheels for CPython 3.8 to 3.12 only.  On newer
interpreters pip falls back to the sdist, which fails on Windows before
anything compiles because its ``pyproject.toml``

1. lists ``patchelf`` (a Linux-only ELF tool with no Windows build) as an
   unconditional build requirement, and
2. spells ``license-files`` as a table, which current maturin rejects.

The Rust extension itself compiles fine.  This script downloads the sdist,
patches those two lines, builds a wheel for the running interpreter and
installs it.  maturin fetches a temporary Rust toolchain on its own; the
only thing you need locally is the MSVC linker (Visual Studio Build Tools
with the "Desktop development with C++" workload).

Usage
-----
    python tools/build_aisdb.py                 # build + install aisdb 1.7.2
    python tools/build_aisdb.py --version 1.7.2 --out dist/wheels
    python tools/build_aisdb.py --no-install    # just produce the wheel

Run it with the interpreter you want aisdb installed into (activate the
venv first).  If PyPI already has a wheel for that interpreter the script
installs it and exits without building anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

DEFAULT_VERSION = "1.7.2"
PYPI_JSON = "https://pypi.org/pypi/aisdb/{version}/json"

# Environment variables that make Python report a different interpreter
# than the one actually running (OSGeo4W shells set all three).  pyo3 and
# maturin introspect ``sys.executable``, so they must not leak into the build.
_INTERPRETER_OVERRIDES = ("PYTHONEXECUTABLE", "PYTHONHOME", "PYTHONPATH")


def _real_interpreter() -> str:
    """``sys.executable`` unless an override variable has replaced it with a shim."""
    if not any(k in os.environ for k in _INTERPRETER_OVERRIDES):
        return sys.executable
    bindir = "Scripts" if os.name == "nt" else "bin"
    exe = "python.exe" if os.name == "nt" else "python"
    # An activated venv is the strongest signal: with the override set Python
    # cannot even tell it is running inside one (sys.prefix falls back to the
    # base install), so ``VIRTUAL_ENV`` must be checked first.
    roots = [Path(v) for v in (os.environ.get("VIRTUAL_ENV"),) if v] + [Path(sys.prefix)]
    for root in roots:
        for candidate in (root / bindir / exe, root / exe):
            if candidate.exists():
                print(
                    f"note: {'/'.join(k for k in _INTERPRETER_OVERRIDES if k in os.environ)}"
                    f" is set; using {candidate} instead of sys.executable ({sys.executable})",
                    flush=True,
                )
                return str(candidate)
    return sys.executable


PYTHON = _real_interpreter()


def _clean_env() -> dict[str, str]:
    """Environment for child processes with the interpreter overrides removed.

    With ``PYTHONEXECUTABLE`` set, even a venv's own ``python.exe`` fails to
    recognise its venv and pip installs into the base interpreter instead.
    """
    return {k: v for k, v in os.environ.items() if k not in _INTERPRETER_OVERRIDES}


def _pip(*args: str, env: dict[str, str] | None = None, check: bool = True) -> int:
    cmd = [PYTHON, "-m", "pip", *args]
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, env=_clean_env() if env is None else env, check=check).returncode


def _build_env() -> dict[str, str]:
    env = _clean_env()
    # aisdb pins pyo3 0.20, which officially knows Pythons up to 3.12.  This
    # flag lets it target newer interpreters through the stable ABI.
    env.setdefault("PYO3_USE_ABI3_FORWARD_COMPATIBILITY", "1")
    env.setdefault("PYO3_PYTHON", PYTHON)
    return env


def try_binary_install(version: str) -> bool:
    """Return True if PyPI already had a wheel for this interpreter."""
    rc = _pip("install", "--quiet", "--only-binary", "aisdb", f"aisdb=={version}", check=False)
    if rc != 0:
        print(f"no PyPI wheel for aisdb {version} on this interpreter; building from source")
    return rc == 0


def download_sdist(version: str, dest: Path) -> Path:
    with urllib.request.urlopen(PYPI_JSON.format(version=version), timeout=60) as r:
        meta = json.load(r)
    sdists = [u for u in meta["urls"] if u["packagetype"] == "sdist"]
    if not sdists:
        raise SystemExit(f"aisdb {version}: no sdist on PyPI")
    info = sdists[0]
    target = dest / info["filename"]
    print(f"downloading {info['url']}", flush=True)
    with urllib.request.urlopen(info["url"], timeout=300) as r, target.open("wb") as f:
        shutil.copyfileobj(r, f)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != info["digests"]["sha256"]:
        raise SystemExit(f"sha256 mismatch for {target.name}")
    return target


def patch_pyproject(src_dir: Path) -> None:
    pyproject = src_dir / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    original = text

    # 1. patchelf is only meaningful (and only installable) on Linux.
    text, n = re.subn(
        r'"patchelf"(\s*,)',
        lambda m: "\"patchelf; sys_platform == 'linux'\"" + m.group(1),
        text,
        count=1,
    )
    if n == 0:
        print("note: patchelf requirement not found; upstream may have fixed it", flush=True)

    # 2. ``[project.license-files] paths = [...]`` -> ``license-files = [...]``
    #    inside ``[project]`` (PEP 639 shape that maturin >= 1.7 enforces).
    m = re.search(r"\[project\.license-files\]\s*\npaths\s*=\s*(\[[^\]]*\])\s*\n", text)
    if m:
        text = text[: m.start()] + text[m.end() :]
        text = re.sub(
            r"(\[project\]\n)",
            rf"\1license-files = {m.group(1)}\n",
            text,
            count=1,
        )
    else:
        print("note: license-files table not found; upstream may have fixed it", flush=True)

    if text != original:
        pyproject.write_text(text, encoding="utf-8")
        print(f"patched {pyproject}", flush=True)


def build_wheel(src_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    before = set(out_dir.glob("aisdb-*.whl"))
    try:
        _pip("wheel", "--no-deps", "-w", str(out_dir), str(src_dir), env=_build_env())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "aisdb wheel build failed.  On Windows the usual cause is a missing "
            "MSVC linker: install Visual Studio Build Tools with the "
            "'Desktop development with C++' workload and retry."
        ) from exc
    new = set(out_dir.glob("aisdb-*.whl")) - before
    if not new:
        raise SystemExit("pip wheel reported success but produced no aisdb wheel")
    return new.pop()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", default=DEFAULT_VERSION, help="aisdb release to build")
    ap.add_argument(
        "--out", type=Path, default=Path("dist") / "wheels", help="where to put the wheel"
    )
    ap.add_argument("--no-install", action="store_true", help="build only, do not pip install")
    ap.add_argument(
        "--force-build", action="store_true", help="skip the PyPI wheel check and build anyway"
    )
    args = ap.parse_args(argv)

    if not args.force_build and try_binary_install(args.version):
        print(f"aisdb {args.version}: PyPI wheel installed, nothing to build")
        return 0

    with tempfile.TemporaryDirectory(prefix="aisdb-build-") as tmp:
        tmp_path = Path(tmp)
        sdist = download_sdist(args.version, tmp_path)
        with tarfile.open(sdist) as tar:
            tar.extractall(tmp_path, filter="data")
        (src_dir,) = [p for p in tmp_path.iterdir() if p.is_dir()]
        patch_pyproject(src_dir)
        wheel = build_wheel(src_dir, args.out.resolve())

    print(f"built {wheel}")
    if not args.no_install:
        _pip("install", str(wheel))
    return 0


if __name__ == "__main__":
    sys.exit(main())
