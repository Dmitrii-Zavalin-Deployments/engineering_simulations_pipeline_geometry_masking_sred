# tests/unit/conftest.py

import types
import pytest
import src.gmsh_runner as gmsh_runner

class DummyGmshModel:
    def __init__(self, bbox=(0,0,0,10,10,10), inside_points=None, entities_ret=None):
        self._bbox = bbox
        self._inside_points = inside_points or set()
        self._entities_ret = entities_ret
    def add(self, name): pass
    def getEntities(self, dim):
        return self._entities_ret if self._entities_ret is not None else [(3,1)]
    def getBoundingBox(self, dim, tag): return self._bbox
    def isInside(self, dim, tag, coords):
        return tuple(round(c,3) for c in coords) in self._inside_points

class DummyGmsh:
    def __init__(self, bbox=(0,0,0,10,10,10), inside_points=None, entities_ret=None):
        self.model = DummyGmshModel(bbox, inside_points, entities_ret)
        self.logger = types.SimpleNamespace(start=lambda: None)
        self._finalized = False
    def initialize(self): pass
    def finalize(self): self._finalized = True
    def open(self, path): pass



