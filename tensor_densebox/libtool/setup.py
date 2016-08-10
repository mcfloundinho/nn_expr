from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import os

os.environ["CXX"] = "g++-4.9"

ext_modules = [Extension('nms',
    sources=['nms_impl.pyx'],
    language='c++',
    extra_compile_args='-std=c++11 -O0'.split(),
    extra_link_args='-lstdc++'.split(),
    include_dirs = [],
    library_dirs = [],
    libraries = [],
    extra_objects=['libnms.a'],
    )]

setup(name='nms', cmdclass={'build_ext': build_ext}, ext_modules=ext_modules)
