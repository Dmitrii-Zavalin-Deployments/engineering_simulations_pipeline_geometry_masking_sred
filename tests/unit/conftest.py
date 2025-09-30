import pytest
import sys

class DummyGmsh:
    """A mock Gmsh class to simulate the gmsh library's behavior for testing."""
    def __init__(self):
        self.initialized = False
        self.finalized = False
        self.options = DummyGmshOptions()
        self.model = DummyGmshModel()
        self.on_progress = None

    def initialize(self, *args, **kwargs):
        self.initialized = True
        print("Mock Gmsh initialized.")

    def finalize(self):
        self.finalized = True
        print("Mock Gmsh finalized.")

    def write(self, filename):
        print(f"Mock Gmsh writing to {filename}")

class DummyGmshOptions:
    """A mock for Gmsh options."""
    def setNumber(self, option, value):
        print(f"Mock Gmsh setNumber called with {option}={value}")
    def setString(self, option, value):
        print(f"Mock Gmsh setString called with {option}={value}")

class DummyGmshModel:
    """A mock for Gmsh model operations."""
    def open(self, path):
        print(f"Mock Gmsh model open called for {path}")
    def getEntities(self, dim):
        if dim == 3:
            return [(3, 1)]  # A single 3D volume
        return []

@pytest.fixture(scope="session")
def dummy_gmsh_instance():
    """Provides a dummy Gmsh instance for mocking."""
    # Ensure gmsh is not already in sys.modules to prevent conflicts
    sys.modules['gmsh'] = DummyGmsh()
    yield
    # Clean up the mock
    del sys.modules['gmsh']

@pytest.fixture(scope="function")
def gmsh_finalized_checker(dummy_gmsh_instance):
    """A fixture to check if Gmsh is properly finalized after a test."""
    yield
    assert sys.modules['gmsh'].finalized is True
