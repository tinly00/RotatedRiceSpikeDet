"""
    setup.py file for SWIG example
"""
from distutils.core import setup, Extension
import numpy
import os  #daaize
from distutils.sysconfig import get_config_vars  #daaize

polyiou_module = Extension('_polyiou',
                           sources=['polyiou_wrap.cxx', 'polyiou.cpp'],
                           )
setup(name = 'polyiou',
      version = '0.1',
      author = "SWIG Docs",
      description = """Simple swig example from docs""",
      ext_modules = [polyiou_module],
      py_modules = ["polyiou"],
)

### daaize
(opt,) = get_config_vars('OPT')
os.environ['OPT'] = " ".join(
    flag for flag in opt.split() if flag != '-Wstrict-prototypes'
)
### daaize
