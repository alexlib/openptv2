#!/usr/bin/env python3
"""Generate a setup.py + pyproject.toml for compiling a pure-mode Cython
module.

Usage:
    python build_template.py mymodule.py [--numpy] [--openmp] [--out-dir DIR]

Writes setup.py and pyproject.toml into --out-dir (default: current
directory). Review the generated files before building — in particular the
module name, and whether you need to add extra include_dirs / libraries for
your own C dependencies.

Build with:
    pip install -e .
or:
    python setup.py build_ext --inplace
"""
import argparse
import os
import sys

SETUP_TEMPLATE = """\
from setuptools import setup
from Cython.Build import cythonize
{numpy_import}

extensions = [
    "{module_path}",
]

setup(
    name="{package_name}",
    ext_modules=cythonize(
        extensions,
        compiler_directives={{"language_level": "3"}},
        annotate=True,  # writes a .html file showing Python-vs-C interaction per line
    ),
{numpy_include}
)
"""

# Note: cythonize() accepts either bare filenames (simple case) or a list of
# setuptools.Extension objects (needed for extra_compile_args / openmp /
# numpy include dirs). We switch templates depending on flags.

SETUP_TEMPLATE_EXTENSION = """\
from setuptools import setup, Extension
from Cython.Build import cythonize
{numpy_import}

ext_modules = [
    Extension(
        "{module_name}",
        ["{module_path}"],
{extra_compile_args}{extra_link_args}{include_dirs}
    ),
]

setup(
    name="{package_name}",
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={{"language_level": "3"}},
        annotate=True,  # writes a .html file showing Python-vs-C interaction per line
    ),
)
"""

PYPROJECT_TEMPLATE = """\
[build-system]
requires = ["setuptools>=61", "Cython>=3.0", {numpy_requires}]
build-backend = "setuptools.build_meta"
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", help="Path to the pure-mode .py file to compile")
    parser.add_argument(
        "--numpy",
        action="store_true",
        help="Module uses NumPy / typed memoryviews backed by NumPy arrays",
    )
    parser.add_argument(
        "--openmp",
        action="store_true",
        help="Module uses cython.parallel.prange and needs OpenMP compiler flags",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Directory to write setup.py / pyproject.toml into (default: cwd)",
    )
    args = parser.parse_args()

    module_path = args.module
    if not module_path.endswith(".py"):
        print("warning: expected a .py file", file=sys.stderr)

    module_name = os.path.splitext(os.path.basename(module_path))[0]
    package_name = module_name.replace("_", "-")

    numpy_import = "import numpy\n" if args.numpy else ""
    numpy_include = (
        "    include_dirs=[numpy.get_include()],\n" if args.numpy else ""
    )
    numpy_requires = '"numpy"' if args.numpy else ""

    if args.openmp:
        extra_compile_args = (
            "        extra_compile_args=['-fopenmp'],  # use '/openmp' on MSVC\n"
        )
        extra_link_args = (
            "        extra_link_args=['-fopenmp'],  # omit on MSVC\n"
        )
        include_dirs = (
            "        include_dirs=[numpy.get_include()],\n" if args.numpy else ""
        )
        setup_content = SETUP_TEMPLATE_EXTENSION.format(
            numpy_import=numpy_import,
            module_name=module_name,
            module_path=module_path,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            include_dirs=include_dirs,
            package_name=package_name,
        )
    elif args.numpy:
        # Simple bare-filename form still works for cythonize, but we need
        # include_dirs for numpy headers, so use the Extension form too.
        setup_content = SETUP_TEMPLATE_EXTENSION.format(
            numpy_import=numpy_import,
            module_name=module_name,
            module_path=module_path,
            extra_compile_args="",
            extra_link_args="",
            include_dirs="        include_dirs=[numpy.get_include()],\n",
            package_name=package_name,
        )
    else:
        setup_content = SETUP_TEMPLATE.format(
            numpy_import=numpy_import,
            module_path=module_path,
            package_name=package_name,
            numpy_include="",
        )

    pyproject_content = PYPROJECT_TEMPLATE.format(numpy_requires=numpy_requires)
    # Clean up trailing comma artifacts if numpy_requires is empty
    pyproject_content = pyproject_content.replace(', ]', ']').replace('[build-system]\nrequires = ["setuptools>=61", "Cython>=3.0", ]', '[build-system]\nrequires = ["setuptools>=61", "Cython>=3.0"]')

    os.makedirs(args.out_dir, exist_ok=True)
    setup_path = os.path.join(args.out_dir, "setup.py")
    pyproject_path = os.path.join(args.out_dir, "pyproject.toml")

    with open(setup_path, "w") as f:
        f.write(setup_content)
    with open(pyproject_path, "w") as f:
        f.write(pyproject_content)

    print(f"Wrote {setup_path}")
    print(f"Wrote {pyproject_path}")
    print()
    print("Next steps:")
    print(f"  1. Review {setup_path} (module name, include dirs, flags).")
    print("  2. Build: pip install -e .   (or: python setup.py build_ext --inplace)")
    print(
        "  3. Open the generated <module>.html annotate file and look for yellow "
        "lines in your hot function — yellow means Python-level interaction "
        "Cython couldn't optimize away."
    )


if __name__ == "__main__":
    main()
