import pytest

@pytest.fixture(autouse=True)
def mock_gmsh(monkeypatch):
    monkeypatch.setattr("gmsh.initialize", lambda: None)
    monkeypatch.setattr("gmsh.finalize", lambda: None)
    monkeypatch.setattr("gmsh.isInitialized", lambda: True)
    monkeypatch.setattr("gmsh.model.getEntities", lambda dim: [(3, 1)])
    monkeypatch.setattr("gmsh.model.getBoundingBox", lambda dim, tag: [0, 0, 0, 1, 1, 1])
    monkeypatch.setattr("gmsh.model.isInside", lambda dim, tag, pt: True)

@pytest.fixture
def mock_dropbox(monkeypatch):
    class MockFileMetadata:
        def __init__(self, name):
            self.name = name
            self.path_lower = f"/mock/{name}"

    class MockResponse:
        def __init__(self, content):
            self.content = content

    class MockDropbox:
        def __init__(self, token):
            self.token = token
            self.counter = 0

        def files_list_folder(self, folder):
            return self._mock_result()

        def files_list_folder_continue(self, cursor):
            return self._mock_result()

        def _mock_result(self):
            self.counter += 1
            return mock.Mock(
                entries=[
                    MockFileMetadata("model.step"),
                    MockFileMetadata("notes.txt"),
                    MockFileMetadata("config.json")
                ],
                has_more=self.counter < 2,
                cursor="mock_cursor"
            )

        def files_download(self, path):
            return None, MockResponse(b"file content")

    monkeypatch.setattr("dropbox.Dropbox", MockDropbox)
