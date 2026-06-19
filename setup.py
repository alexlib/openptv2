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
from pathlib import Path

from setuptools import setup, Extension, Command, find_namespace_packages
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


def _ensure_include_structure():
    """Ensure lib/include/optv/ exists with headers for Cython includes."""
    optv_inc = LIB_INC / "optv"
    if not optv_inc.exists():
        optv_inc.mkdir(parents=True)
        for hdr in LIB_INC.glob("*.h"):
            dst = optv_inc / hdr.name
            if not dst.exists():
                shutil.copy2(hdr, dst)


def _cythonize_all():
    """Run Cython on all .pyx files in bindings/optv/."""
    print("[OpenPTV2] Starting Cythonization of bindings...")
    _ensure_include_structure()
    pyx_files = sorted(BINDINGS_OPTV.glob("*.pyx"))
    if not pyx_files:
        print("[OpenPTV2] WARNING: No .pyx files found to cythonize.")
        return

    from Cython.Build import cythonize

    cythonize(
        [str(p) for p in pyx_files],
        compiler_directives={"language_level": "3"},
        include_path=[str(BINDINGS), str(BINDINGS_OPTV), str(LIB_INC)],
    )
    print("[OpenPTV2] Cythonization completed successfully.")


def _needs_rebuild():
    """Check if C sources or Cython files changed since last build."""
    if os.environ.get("OPENPTV_PYTHON_ONLY"):
        print("[OpenPTV2] OPENPTV_PYTHON_ONLY is set. Skipping compilation.")
        return False
    c_files = list(BINDINGS_OPTV.glob("*.c"))
    if not c_files:
        print("[OpenPTV2] No existing compiled C files found. Rebuild required.")
        return True
    for pyx in BINDINGS_OPTV.glob("*.pyx"):
        c_file = pyx.with_suffix(".c")
        if not c_file.exists() or pyx.stat().st_mtime > c_file.stat().st_mtime:
            print(f"[OpenPTV2] Cython file modified: {pyx.name}. Rebuild required.")
            return True
    for src in LIB_SRC.glob("*.c"):
        c_file = src.with_suffix(".c")
        if not c_file.exists() or src.stat().st_mtime > c_file.stat().st_mtime:
            print(f"[OpenPTV2] C library source modified: {src.name}. Rebuild required.")
            return True
    return False


# ---------------------------------------------------------------------------
# Prepare Cython files before setup() runs
# ---------------------------------------------------------------------------
if _needs_rebuild():
    _cythonize_all()


# ---------------------------------------------------------------------------
# Extension building
# ---------------------------------------------------------------------------
def get_liboptv_sources():
    """Get all C source files from lib/src/."""
    if os.environ.get("OPENPTV_PYTHON_ONLY"):
        return []
    return [str(f.relative_to(ROOT)) for f in sorted(LIB_SRC.glob("*.c"))]


def mk_ext(name, cython_c_file):
    """Create a setuptools Extension for one Cython module + the C library."""
    include_dirs = [
        numpy.get_include(),
        str(LIB_INC.relative_to(ROOT)),
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
    if os.environ.get("OPENPTV_PYTHON_ONLY"):
        return []
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

    description = "Run Cython"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        _cythonize_all()


class BuildExtWithPrepare(build_ext):
    """Custom build_ext that runs Cython before compiling."""

    def finalize_options(self):
        super().finalize_options()
        # Parallel extension builds race on shared lib/src/*.c object paths on
        # Windows, which causes fatal C1083 permission errors in cibuildwheel.
        if sys.platform.startswith("win"):
            self.parallel = None
        else:
            # Enable parallel C compilation to speed up builds on multi-core CPUs
            if not self.parallel:
                self.parallel = os.cpu_count() or 1
                print(f"[OpenPTV2] Parallel compilation enabled: compiling with {self.parallel} jobs.")

    def run(self):
        if _needs_rebuild():
            _cythonize_all()
        print("[OpenPTV2] Compiling and linking C/Cython extensions...")
        super().run()
        print("[OpenPTV2] C/Cython extensions built successfully!")


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
        packages=find_namespace_packages(
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
        package_dir={
            "optv": "bindings/optv",
            "gui.pyptv": "gui/pyptv",
            "gui.plugins": "gui/plugins",
        },
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
