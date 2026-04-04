# -*- coding: utf-8 -*-
"""
Unified build script for openptv2.

Builds the C library, Cython bindings, and installs all Python packages
(algorithms, openptv2, gui) in one step.

Usage:
    pip install -e .          # Development install
    pip install .             # Regular install
"""

import os
import shutil
import sys
import glob
import subprocess
from pathlib import Path

from setuptools import setup, Extension, Command, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install

import numpy

# ---------------------------------------------------------------------------
# Paths — always relative to this setup.py file (project root)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.resolve()
LIB_SRC = ROOT / "lib" / "src"
LIB_INC = ROOT / "lib" / "include"
BINDINGS = ROOT / "bindings"
BINDINGS_OPTV = BINDINGS / "optv"
LIBOPTV = BINDINGS / "liboptv"
LIBOPTV_SRC = LIBOPTV / "src"
LIBOPTV_INC = LIBOPTV / "include"


def _ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def _copy_sources():
    """Copy C sources/headers from lib/ into bindings/liboptv/."""
    _ensure_dir(LIBOPTV_SRC)
    _ensure_dir(LIBOPTV_INC)
    _ensure_dir(LIBOPTV_INC / "optv")

    for src in LIB_SRC.glob("*.c"):
        dst = LIBOPTV_SRC / src.name
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(str(src), str(dst))

    for hdr in LIB_INC.glob("*.h"):
        dst = LIBOPTV_INC / hdr.name
        if not dst.exists():
            shutil.copy2(str(hdr), str(dst))
        dst_optv = LIBOPTV_INC / "optv" / hdr.name
        if not dst_optv.exists():
            shutil.copy2(str(hdr), str(dst_optv))

    optv_sub = LIB_INC / "optv"
    if optv_sub.is_dir():
        for hdr in optv_sub.glob("*.h"):
            dst = LIBOPTV_INC / "optv" / hdr.name
            if not dst.exists():
                shutil.copy2(str(hdr), str(dst))


def _cythonize_all():
    """Run Cython on all .pyx files in bindings/optv/."""
    pyx_files = sorted(BINDINGS_OPTV.glob("*.pyx"))
    if not pyx_files:
        return

    from Cython.Build import cythonize

    cythonize(
        [str(p) for p in pyx_files],
        compiler_directives={"language_level": "3"},
        include_path=[str(BINDINGS), str(BINDINGS_OPTV)],
    )


def _needs_rebuild():
    """Check if C sources or Cython files changed since last build."""
    c_files = list(BINDINGS_OPTV.glob("*.c"))
    if not c_files:
        return True
    for pyx in BINDINGS_OPTV.glob("*.pyx"):
        c_file = pyx.with_suffix(".c")
        if not c_file.exists() or pyx.stat().st_mtime > c_file.stat().st_mtime:
            return True
    for src in LIB_SRC.glob("*.c"):
        dst = LIBOPTV_SRC / src.name
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            return True
    return False


# ---------------------------------------------------------------------------
# Prepare sources immediately (before setup() is called)
# This runs in the build backend's working directory (project root).
# ---------------------------------------------------------------------------
_copy_sources()
if _needs_rebuild():
    _cythonize_all()


# ---------------------------------------------------------------------------
# Extension building
# ---------------------------------------------------------------------------
def get_liboptv_sources():
    """Get all C source files from the bundled liboptv directory."""
    return [str(f.relative_to(ROOT)) for f in sorted(LIBOPTV_SRC.glob("*.c"))]


def mk_ext(name, cython_c_file):
    """Create a setuptools Extension for one Cython module + the C library."""
    include_dirs = [
        numpy.get_include(),
        str(LIBOPTV_INC.relative_to(ROOT)),
        str(BINDINGS_OPTV.relative_to(ROOT)),
    ]

    extra_compile_args = []
    extra_link_args = []
    if not sys.platform.startswith("win"):
        extra_compile_args.extend(["-Wno-cpp", "-Wno-unused-function"])
        extra_link_args.extend(["-Wl,-rpath,$ORIGIN"])
    else:
        extra_compile_args.extend(["/W4", "/std:c11", "/D_CRT_SECURE_NO_WARNINGS"])

    # Use relative paths — required for isolated builds (python -m build)
    all_sources = [str(cython_c_file.relative_to(ROOT))] + get_liboptv_sources()
    # Verify all source files exist before creating extension
    for src in all_sources:
        if not (ROOT / src).exists():
            raise FileNotFoundError(
                f"Source file not found: {src}\n"
                f"Run 'python setup.py prepare' first, or use 'pip install -e .' "
                f"which triggers preparation automatically."
            )

    return Extension(
        name,
        all_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )


def get_extensions():
    """Create Extension objects for all Cython modules."""
    extensions = []
    for pyx in sorted(BINDINGS_OPTV.glob("*.pyx")):
        c_file = pyx.with_suffix(".c")
        module_name = f"optv.{pyx.stem}"
        extensions.append(mk_ext(module_name, c_file))
    return extensions


# ---------------------------------------------------------------------------
# Custom commands
# ---------------------------------------------------------------------------
class PrepareSources(Command):
    """Prepare C sources and run Cython."""

    description = "Copy C sources and run Cython"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        _copy_sources()
        _cythonize_all()


class BuildExtWithPrepare(build_ext):
    """Custom build_ext that prepares sources before compiling."""

    def run(self):
        _copy_sources()
        if _needs_rebuild():
            _cythonize_all()
        super().run()


class BuildPyWithExtensions(build_py):
    """Ensure extensions are built before pure Python packages."""

    def run(self):
        self.run_command("build_ext")
        super().run()


class DevelopWithExtensions(develop):
    """Editable install with extension building."""

    def run(self):
        self.run_command("build_ext")
        super().run()


class InstallWithExtensions(install):
    """Regular install with extension building."""

    def run(self):
        self.run_command("build_ext")
        super().run()


# ---------------------------------------------------------------------------
# Main setup
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    setup(
        packages=find_packages(
            include=[
                "openptv2",
                "openptv2.*",
                "algorithms",
                "algorithms.*",
                "gui.pyptv",
                "gui.pyptv.*",
                "gui.plugins",
                "gui.plugins.*",
                "optv",
                "optv.*",
            ],
        ),
        package_dir={"optv": "bindings/optv"},
        ext_modules=get_extensions(),
        cmdclass={
            "build_ext": BuildExtWithPrepare,
            "build_py": BuildPyWithExtensions,
            "develop": DevelopWithExtensions,
            "install": InstallWithExtensions,
            "prepare": PrepareSources,
        },
        include_package_data=True,
        package_data={
            "optv": ["*.pxd", "*.pyx", "optv/*.h"],
        },
        zip_safe=False,
    )
