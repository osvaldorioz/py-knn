from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import numpy

ext_modules = [
    Pybind11Extension(
        "lda_cpp",
        ["lda2.cpp"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-std=c++11'],
    ),
]

setup(
    name="lda_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
