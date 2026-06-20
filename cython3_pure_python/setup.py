from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "structures",
        sources=["structures.py"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="structures",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)
