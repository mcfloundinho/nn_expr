import pyximport
importers = pyximport.install()
from .nms_impl import *
pyximport.uninstall(*importers)
