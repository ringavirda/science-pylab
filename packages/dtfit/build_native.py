"""Compile the dtfit._core._native C extension with clang (standalone dev builder).

The package works in pure Python without this step; building it swaps the
compiled numeric kernels in for the SciPy/NumPy fallbacks (see
``dtfit/_kernels.py``) and speeds up the integral-based methods. It is a raw
CPython + NumPy C-API extension compiled directly with clang -- no setuptools /
pybind11 required, which keeps it working on bleeding-edge Python.

``pip install`` now builds the same extension automatically via the optional
``Extension`` declared in ``setup.py`` (``optional=True`` -> graceful NumPy
fallback when no compiler is present). This script remains for the fast dev loop:
an incremental clang rebuild without a full reinstall, plus the VS Code
IntelliSense config and ``--clean``.

Cross-platform build:

* **Windows.** CPython is built with the MSVC ABI, so the extension is compiled
  with ``clang-cl`` (clang's MSVC-compatible driver) inside the Visual Studio
  developer environment, which supplies the Windows SDK / CRT headers and import
  libraries. The VS install is located with ``vswhere`` and its
  ``vcvars64.bat`` is sourced before invoking clang-cl.
* **Linux / macOS.** Plain ``clang -shared -fPIC`` against the interpreter's
  headers; libpython is not linked (extension symbols are resolved by the
  interpreter at import). macOS adds ``-undefined dynamic_lookup`` for the same
  reason.

Usage:
    python build_native.py            # build into src/dtfit/_core/
    python build_native.py --clean    # remove the built artifacts
    python build_native.py --ide      # (re)write .vscode/c_cpp_properties.json

A normal build also refreshes the VS Code C/C++ IntelliSense config so the
editor can resolve ``Python.h`` and the NumPy headers.

clang / clang-cl is expected on PATH; otherwise the standard Windows LLVM
install location (``C:/Program Files/LLVM/bin``) is tried automatically.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# The native extension lives in the private numeric core (dtfit._core._native),
# next to the _kernels wrapper that loads it.
PKG = ROOT / "src" / "dtfit" / "_core"
SRC = PKG / "_native.c"
LLVM_BIN = Path(r"C:/Program Files/LLVM/bin")


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _which(name: str) -> str:
    """Locate an executable: PATH first, then the Windows LLVM default dir."""
    exe = shutil.which(name)
    if exe:
        return exe
    candidate = LLVM_BIN / f"{name}.exe"
    if candidate.exists():
        return str(candidate)
    raise SystemExit(
        f"{name} not found on PATH"
        + (f" or in {LLVM_BIN}" if os.name == "nt" else "")
        + ". Install LLVM/clang or add it to PATH."
    )


def includes() -> tuple[str, str]:
    import numpy

    return sysconfig.get_path("include"), numpy.get_include()


def output_path() -> Path:
    """Extension filename with the interpreter's ABI suffix (importable as
    ``dtfit._core._native``)."""
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or (
        ".pyd" if os.name == "nt" else ".so"
    )
    return PKG / f"_native{suffix}"


def clean() -> None:
    removed = []
    for p in PKG.glob("_native*"):
        if p.suffix.lower() in {
            ".pyd", ".so", ".dylib", ".obj", ".o", ".exp", ".lib", ".pdb"
        }:
            p.unlink()
            removed.append(p.name)
    print("Removed:", ", ".join(removed) if removed else "(nothing)")


# --------------------------------------------------------------------------- #
# IDE config: let VS Code IntelliSense resolve Python.h / NumPy headers
# --------------------------------------------------------------------------- #
def write_ide_config() -> None:
    """Generate .vscode/c_cpp_properties.json with the active include paths.

    The paths are interpreter/machine specific, so this is regenerated locally
    rather than committed (it is git-ignored).
    """
    py_include, np_include = includes()
    if os.name == "nt":
        compiler, mode = _which("clang-cl"), "windows-clang-x64"
    elif sys.platform == "darwin":
        compiler, mode = _which("clang"), "macos-clang-x64"
    else:
        compiler, mode = _which("clang"), "linux-clang-x64"

    config = {
        "version": 4,
        "configurations": [
            {
                "name": "dtfit-native",
                "includePath": [
                    "${workspaceFolder}/**",
                    py_include,
                    np_include,
                ],
                "defines": [],
                "compilerPath": compiler,
                "cStandard": "c11",
                "intelliSenseMode": mode,
            }
        ],
    }
    vscode = ROOT / ".vscode"
    vscode.mkdir(exist_ok=True)
    dest = vscode / "c_cpp_properties.json"
    dest.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print("Wrote", dest.relative_to(ROOT))


# --------------------------------------------------------------------------- #
# platform builds
# --------------------------------------------------------------------------- #
def _find_vcvars() -> Path:
    """Locate the VS developer-environment batch script via vswhere."""
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:/Program Files (x86)")
    vswhere = Path(pf86) / "Microsoft Visual Studio/Installer/vswhere.exe"
    if not vswhere.exists():
        raise SystemExit(
            "vswhere not found; install Visual Studio Build Tools (with the "
            "C++ workload and a Windows SDK)."
        )
    out = subprocess.run(
        [str(vswhere), "-products", "*", "-latest",
         "-property", "installationPath"],
        capture_output=True, text=True,
    ).stdout.strip()
    if not out:
        # -latest can skip the BuildTools product id; take the first of -all.
        lines = subprocess.run(
            [str(vswhere), "-products", "*", "-all",
             "-property", "installationPath"],
            capture_output=True, text=True,
        ).stdout.splitlines()
        out = lines[0].strip() if lines else ""
    if not out:
        raise SystemExit("No Visual Studio / Build Tools installation found.")
    vcvars = Path(out) / "VC/Auxiliary/Build/vcvars64.bat"
    if not vcvars.exists():
        raise SystemExit(f"vcvars64.bat not found under {out}.")
    return vcvars


def _build_windows() -> None:
    clang_cl = _which("clang-cl")
    vcvars = _find_vcvars()
    out = output_path()
    py_include, np_include = includes()
    libs_dir = Path(sysconfig.get_config_var("installed_base")) / "libs"
    obj = PKG / "_native.obj"

    # pyconfig.h emits `#pragma comment(lib, "pythonXY.lib")` under MSVC, so the
    # Python import library is requested automatically -- we only supply its dir.
    compile_cmd = (
        f'"{clang_cl}" /nologo /O2 /DNDEBUG /LD '
        f'/I"{py_include}" /I"{np_include}" '
        f'"{SRC}" /Fo"{obj}" /Fe:"{out}" '
        f'/link /LIBPATH:"{libs_dir}"'
    )
    # Source vcvars64 first so the Windows SDK / CRT headers and libs are found.
    script = f'@echo off\r\ncall "{vcvars}" >nul\r\n{compile_cmd}\r\n'

    print("clang-cl:", clang_cl)
    print("vcvars:", vcvars)
    print("output:", out)

    with tempfile.NamedTemporaryFile(
        "w", suffix=".bat", delete=False, dir=ROOT
    ) as fh:
        fh.write(script)
        bat = fh.name
    try:
        res = subprocess.run(["cmd", "/c", bat], cwd=ROOT)
    finally:
        os.unlink(bat)
    if res.returncode != 0:
        raise SystemExit(f"clang-cl failed with exit code {res.returncode}")

    # Tidy the intermediate object and the import .lib/.exp left next to the .pyd.
    for junk in (obj, out.with_suffix(".lib"), out.with_suffix(".exp")):
        if junk.exists():
            junk.unlink()
    print("Built", out.name)


def _build_posix() -> None:
    clang = _which("clang")
    out = output_path()
    py_include, np_include = includes()

    cmd = [
        clang, "-O3", "-shared", "-fPIC", "-DNDEBUG",
        f"-I{py_include}", f"-I{np_include}",
        str(SRC), "-o", str(out),
    ]
    if sys.platform == "darwin":
        # On macOS, leave Python symbols undefined for the interpreter to supply.
        cmd[3:3] = ["-undefined", "dynamic_lookup"]

    print("clang:", clang)
    print("output:", out)
    print("cmd:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=ROOT)
    if res.returncode != 0:
        raise SystemExit(f"clang failed with exit code {res.returncode}")
    print("Built", out.name)


def build() -> None:
    if os.name == "nt":
        _build_windows()
    else:
        _build_posix()
    # Keep the editor's IntelliSense config in sync with the build.
    write_ide_config()


if __name__ == "__main__":
    if "--clean" in sys.argv:
        clean()
    elif "--ide" in sys.argv:
        write_ide_config()
    else:
        build()
