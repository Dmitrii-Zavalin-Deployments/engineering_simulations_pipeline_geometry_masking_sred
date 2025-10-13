import dropbox
# tests/test_download_dropbox_files.py

import pytest
import os
import json
from unittest import mock
from src.download_dropbox_files import (
    refresh_access_token,
    download_files_from_dropbox,
    ALLOWED_EXTENSIONS
)

# --- Token Refresh Tests ---

@mock.patch("requests.post")
def test_refresh_access_token_success(mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"access_token": "mock_token"}
    token = refresh_access_token("refresh", "id", "secret")
    assert token == "mock_token"

@mock.patch("requests.post")
def test_refresh_access_token_failure(mock_post):
    mock_post.return_value.status_code = 401
    mock_post.return_value.text = "Unauthorized"
    with pytest.raises(Exception, match="Failed to refresh access token"):
        refresh_access_token("bad", "id", "secret")

# --- Dropbox Download Tests ---

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

@pytest.fixture
def temp_log_and_folder(tmp_path):
    log_file = tmp_path / "log.txt"
    local_folder = tmp_path / "downloads"
    return str(local_folder), str(log_file)

@mock.patch("src.download_dropbox_files.refresh_access_token", return_value="mock_token")
def test_download_files_success(mock_refresh, mock_dropbox, temp_log_and_folder):
    local_folder, log_file = temp_log_and_folder
    download_files_from_dropbox(
        dropbox_folder="/mock_folder",
        local_folder=local_folder,
        refresh_token="refresh",
        client_id="id",
        client_secret="secret",
        log_file_path=log_file
    )

    # Check downloaded files
    downloaded = os.listdir(local_folder)
    assert "model.step" in downloaded
    assert "config.json" in downloaded
    assert "notes.txt" not in downloaded  # Unsupported extension

    # Check log content
    with open(log_file, "r") as f:
        log = f.read()
        assert "✅ Downloaded model.step" in log
        assert "⏭️ Skipped file (unsupported type): notes.txt" in log
        assert "🎉 Download completed." in log

@mock.patch("src.download_dropbox_files.refresh_access_token", return_value="mock_token")
def test_dropbox_api_error(mock_refresh, monkeypatch, temp_log_and_folder):
    class MockDropbox:
        def __init__(self, token):
    raise dropbox.exceptions.ApiError("mock_request", "mock_error", "text", "en")

    monkeypatch.setattr("dropbox.Dropbox", MockDropbox)
    local_folder, log_file = temp_log_and_folder

    download_files_from_dropbox(
        dropbox_folder="/mock_folder",
        local_folder=local_folder,
        refresh_token="refresh",
        client_id="id",
        client_secret="secret",
        log_file_path=log_file
    )

    with open(log_file, "r") as f:
        log = f.read()
        assert "❌ Dropbox API error" in log

@mock.patch("src.download_dropbox_files.refresh_access_token", side_effect=RuntimeError("Simulated failure"))
def test_unexpected_error(mock_refresh, temp_log_and_folder):
    local_folder, log_file = temp_log_and_folder

    download_files_from_dropbox(
        dropbox_folder="/mock_folder",
        local_folder=local_folder,
        refresh_token="refresh",
        client_id="id",
        client_secret="secret",
        log_file_path=log_file
    )

    with open(log_file, "r") as f:
        log = f.read()
        assert "❌ Unexpected error" in log



