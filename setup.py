# -*- coding: utf-8 -*-
"""
Unified build script for openptv2.

Builds and packages only the Cython 3 Pure Python modules in algorithms/
and installs the Python packages used by the unified single-engine runtime.

Usage:
    pip install -e .          # Development install
    pip install .             # Regular install
"""

import os
import shutil
import sys
from pathlib import Path

import numpy
from setuptools import Command, Extension, find_namespace_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install

# ---------------------------------------------------------------------------
# Paths — always relative to this setup.py file (project root)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.resolve()


def _libomp_prefix():
    """libomp prefix for macOS OpenMP (Apple clang needs it).

    Order: explicit OPENPTV_LIBOMP_PREFIX (CI uses a low-deployment-target
    libomp under /usr/local), then `brew --prefix libomp`, then arm64 default.
    """
    import subprocess

    env = os.environ.get("OPENPTV_LIBOMP_PREFIX")
    if env:
        return env
    try:
        return subprocess.check_output(
            ["brew", "--prefix", "libomp"], text=True
        ).strip()
    except Exception:
        return "/opt/homebrew/opt/libomp"


# All 18 modules translated from the C library to Cython 3 Pure Python
# Note: track_kernels.py is a shim re-exporting from sub-modules.
ALGORITHMS_MODULES = [
    "vec_utils",
    "trafo",
    "lsqadj",
    "multimed",
    "ray_tracing",
    "calibration",
    "parameters",
    "imgcoord",
    "orientation",
    "image_processing",
    "segmentation",
    "sortgrid",
    "epi",
    "correspondences",
    "tracking_frame_buf",
    "tracking_run",
    "track",
    "track3d",
    "track_kernels",  # shim re-exporting sub-modules below
    "track_kernels_geom",  # multimedia + pixel_projection + angle
    "track_kernels_search",  # candidate search
    "track_kernels_transform",  # point_position + image_coord transforms
    "track_kernels_pixel",  # per-camera pixel-space math
    "track_kernels_position",  # 3D ray tracing + position reconstruction
    "track_kernels_track3d",  # stereo-3D tracking loop
    "track_kernels_corr",  # forward/backward tracking loops + CAS
    "track_kernels_tracking",  # compatibility shim (re-exports above)
    "track_kernels_batch",  # batch processing + detection
    "track_kernels_hybrid",  # Cython 3 compiled hybrid tracking loop kernel
]


def _cythonize_all():
    """Run Cython on all Pure Python modules in algorithms/."""
    import time

    start_time = time.time()
    print("[OpenPTV2] Starting Cythonization of algorithms pure Python modules...")

    try:
        import Cython
        from Cython.Build import cythonize
    except Exception as e:
        raise RuntimeError(
            "Cython is required for 'python setup.py prepare'. "
            "Install with: python -m pip install 'cython>=3.0.10,<4'"
        ) from e

    from packaging.version import Version

    if Version(Cython.__version__) < Version("3.0.10"):
        raise RuntimeError(
            f"Cython>=3.0.10 required, found {Cython.__version__}. "
            "Upgrade with: python -m pip install 'cython>=3.0.10,<4'"
        )
    import os

    # Collect all existing pure Python modules
    targets = []
    for mod in ALGORITHMS_MODULES:
        py_file = ROOT / "src" / "openptv2" / "algorithms" / f"{mod}.py"
        if py_file.exists():
            targets.append(str(py_file.relative_to(ROOT)))

    # Cythonize all modules in a single call in parallel
    if targets:
        nthreads = os.cpu_count() or 1
        print(
            f"[OpenPTV2] Running cythonize on {len(targets)} targets "
            f"with {nthreads} threads..."
        )
        directives = {
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
            "initializedcheck": False,
        }
        # Opt-in line tracing for coverage (OPENPTV_CYTHON_TRACE=1). Never on by
        # default: line tracing makes the compiled code much slower. Pair with
        # the CYTHON_TRACE_NOGIL macro added in get_extensions().
        if os.environ.get("OPENPTV_CYTHON_TRACE") == "1":
            print("[OpenPTV2] Cython line tracing ENABLED (coverage build, slow)")
            directives["linetrace"] = True
            directives["profile"] = True
        cythonize(
            targets,
            nthreads=nthreads,
            annotate=True,  # generates .html showing Python-vs-C per line
            compiler_directives=directives,
        )
    print(
        f"[OpenPTV2] Cythonization of algorithms completed successfully "
        f"in {time.time() - start_time:.2f} seconds."
    )


def _needs_rebuild():
    """Check if Pure Python modules changed since last build."""
    # Check if pure Python modules need rebuilding
    for mod in ALGORITHMS_MODULES:
        py_file = ROOT / "src" / "openptv2" / "algorithms" / f"{mod}.py"
        if py_file.exists():
            py_c = py_file.with_suffix(".c")
            if not py_c.exists() or py_file.stat().st_mtime >= py_c.stat().st_mtime:
                print(
                    f"[OpenPTV2] Pure Python module modified or uncompiled: {mod}.py. "
                    "Rebuild required."
                )
                return True
    return False


def _setup_ccache_env():
    """Prepend ccache to CC/CXX if available (dramatically speeds up rebuilds)."""
    ccache = shutil.which("ccache")
    if ccache:
        for var in ("CC", "CXX"):
            existing = os.environ.get(var, "")
            if existing:
                os.environ[var] = f"{ccache} {existing}"
            else:
                os.environ[var] = f"{ccache} {var.lower()}"
        print(f"[OpenPTV2] ccache enabled ({ccache})")


def _compiler_available():
    """Best-effort detection of a working C compiler.

    The algorithms/ modules run interpreted as plain Python; compilation only
    adds a speedup. On machines without a toolchain (e.g. Windows without MSVC),
    return False so editable/regular installs fall back to pure Python instead
    of failing the whole build.

    Overrides:
      OPENPTV2_SKIP_COMPILE=1   -> never compile (force pure Python)
      OPENPTV2_FORCE_COMPILE=1  -> always attempt to compile
    """
    if os.environ.get("OPENPTV2_SKIP_COMPILE", "0") in ("1", "true", "True"):
        return False
    if os.environ.get("OPENPTV2_FORCE_COMPILE", "0") in ("1", "true", "True"):
        return True

    try:
        from setuptools._distutils import sysconfig as du_sysconfig
        from setuptools._distutils.ccompiler import new_compiler
    except Exception:
        try:
            from distutils import sysconfig as du_sysconfig
            from distutils.ccompiler import new_compiler
        except Exception:
            return False

    try:
        compiler = new_compiler()
    except Exception:
        return False

    # MSVC: initialize() runs the vcvars lookup that raises when the Build
    # Tools are absent.
    if sys.platform.startswith("win"):
        try:
            compiler.initialize()
            return True
        except Exception:
            return False

    # POSIX: confirm the configured cc executable actually exists on PATH.
    try:
        du_sysconfig.customize_compiler(compiler)
    except Exception:
        pass
    candidates = []
    for attr in ("compiler_so", "compiler", "compiler_cxx"):
        val = getattr(compiler, attr, None)
        if val:
            candidates.append(val[0])
    if not candidates:
        candidates = [os.environ.get("CC", "cc")]
    return any(shutil.which(c) for c in candidates if c)


# Enable ccache for faster rebuilds if available
_setup_ccache_env()

# Resolved once: governs whether C extensions are built or skipped this run.
HAVE_COMPILER = _compiler_available()


# ---------------------------------------------------------------------------
# Prepare Cython files before setup() runs
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not HAVE_COMPILER:
        print(
            "[OpenPTV2] No C compiler detected — installing pure-Python "
            "(interpreted) algorithms. Set OPENPTV2_FORCE_COMPILE=1 to override."
        )
    elif _needs_rebuild():
        _cythonize_all()


def _windows_compile_args(is_dev):
    opt = "/Od" if is_dev else "/O2"
    return [opt, "/W4", "/std:c11", "/D_CRT_SECURE_NO_WARNINGS", "/openmp"], []


def _darwin_compile_args(is_dev):
    # Apple clang rejects bare -fopenmp; it needs Homebrew libomp via
    # -Xpreprocessor -fopenmp + -lomp. delocate bundles libomp.dylib into the
    # wheel and rewrites the rpath.
    opt = "-O0" if is_dev else "-O3"
    libomp = _libomp_prefix()
    compile_args = [
        opt,
        "-Wno-cpp",
        "-Wno-unused-function",
        "-Xpreprocessor",
        "-fopenmp",
        f"-I{libomp}/include",
    ]
    link_args = ["-Wl,-rpath,@loader_path", f"-L{libomp}/lib", "-lomp"]
    return compile_args, link_args


def _posix_compile_args(is_dev):
    opt = "-O0" if is_dev else "-O3"
    compile_args = [
        opt,
        "-Wno-cpp",
        "-Wno-unused-function",
        "-Wno-maybe-uninitialized",
        "-fopenmp",
    ]
    return compile_args, ["-Wl,-rpath,$ORIGIN", "-fopenmp"]


def _platform_compile_args(is_dev):
    """extra_compile_args/extra_link_args for the current platform. Same for every module."""
    if sys.platform.startswith("win"):
        return _windows_compile_args(is_dev)
    if sys.platform == "darwin":
        return _darwin_compile_args(is_dev)
    return _posix_compile_args(is_dev)


def get_extensions():
    """Create Extension objects for all Cython modules."""
    if not HAVE_COMPILER:
        return []

    extensions = []

    # Opt-in coverage build: emit Cython line-trace calls in the generated C.
    # Must match the linetrace directive set in _cythonize_all().
    trace = os.environ.get("OPENPTV_CYTHON_TRACE") == "1"
    # Define both: CYTHON_TRACE enables tracing in gil-holding code, _NOGIL adds
    # nogil sections. NOGIL alone did not force CYTHON_TRACE=1 in the generated C.
    define_macros = (
        [("CYTHON_TRACE", "1"), ("CYTHON_TRACE_NOGIL", "1")] if trace else []
    )

    # Check for fast developer build (O0 / Od)
    is_dev = os.environ.get("DEV_BUILD", "0") in ("1", "true", "True")
    if is_dev:
        print(
            "[OpenPTV2] Fast developer build mode enabled "
            "(using -O0 / /Od compiler flags)"
        )

    extra_compile_args, extra_link_args = _platform_compile_args(is_dev)

    # Cython 3 Pure Python algorithms extensions only
    for mod in ALGORITHMS_MODULES:
        py_file = ROOT / "src" / "openptv2" / "algorithms" / f"{mod}.py"
        if not py_file.exists():
            continue

        c_file = py_file.with_suffix(".c")
        sources = [str(c_file.relative_to(ROOT))]

        # Add CAS shim for MSVC portability (__sync_bool_compare_and_swap)
        if mod == "track_kernels_corr":
            sources.append("src/openptv2/algorithms/cas_shim.c")

        extensions.append(
            Extension(
                f"openptv2.algorithms.{mod}",
                sources=sources,
                include_dirs=[numpy.get_include()],
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                define_macros=define_macros,
            )
        )

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
        try:
            _cythonize_all()
        except Exception as e:
            print(
                "[OpenPTV2] Warning: Cython prepare step failed; "
                f"continuing with available sources. Reason: {e}"
            )


class BuildExtWithPrepare(build_ext):
    """Custom build_ext that runs Cython before compiling."""

    def finalize_options(self):
        super().finalize_options()
        import os

        # We can compile extensions in parallel since generated C sources
        # are pre-generated.
        # and static before compiling begins.
        self.parallel = min(os.cpu_count() or 1, 8)

    def _setup_ccache(self):
        """Prepend ccache to the compiler if available (speeds up rebuilds)."""
        import shutil

        ccache = shutil.which("ccache")
        if ccache and hasattr(self, "compiler"):
            for attr in ("compiler", "compiler_so", "compiler_cxx", "linker_so"):
                cmd = getattr(self.compiler, attr, None)
                if cmd and cmd and cmd[0] != ccache:
                    setattr(self.compiler, attr, [ccache] + cmd)

    def run(self):
        import time

        if not HAVE_COMPILER or not self.extensions:
            print(
                "[OpenPTV2] Skipping Cython extension build — using pure-Python "
                "(interpreted) algorithms."
            )
            return
        start_time = time.time()
        if _needs_rebuild():
            _cythonize_all()
        self._setup_ccache()
        print(
            f"[OpenPTV2] Compiling and linking Cython extensions in "
            f"parallel ({self.parallel} workers)..."
        )
        super().run()
        print(
            f"[OpenPTV2] Cython extensions built successfully in "
            f"{time.time() - start_time:.2f} seconds!"
        )


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
            where="src",
            include=[
                "openptv2",
                "openptv2.*",
            ],
        ),
        package_dir={"": "src"},
        ext_modules=get_extensions(),
        cmdclass={
            "build_ext": BuildExtWithPrepare,
            "build_py": BuildPyWithExtensions,
            "develop": DevelopWithExtensions,
            "install": InstallWithExtensions,
            "prepare": PrepareSources,
        },
        include_package_data=True,
        zip_safe=False,
    )
