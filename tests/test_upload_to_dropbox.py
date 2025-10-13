# tests/test_upload_to_dropbox.py

import pytest
import os
from unittest import mock
from src.upload_to_dropbox import refresh_access_token, upload_file_to_dropbox

# --- refresh_access_token ---

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
    with pytest.raises(Exception, match="Failed to refresh access token: Status Code 401"):
        refresh_access_token("bad", "id", "secret")

# --- upload_file_to_dropbox ---

@pytest.fixture
def temp_file(tmp_path):
    file_path = tmp_path / "test_output.json"
    file_path.write_text('{"key": "value"}')
    return str(file_path)

@mock.patch("src.upload_to_dropbox.refresh_access_token", return_value="mock_token")
@mock.patch("dropbox.Dropbox")
def test_upload_file_success(mock_dbx_class, mock_refresh, temp_file):
    mock_dbx = mock.Mock()
    mock_dbx.files_upload.return_value = None
    mock_dbx_class.return_value = mock_dbx

    result = upload_file_to_dropbox(
        local_file_path=temp_file,
        dropbox_file_path="/mock/test_output.json",
        refresh_token="refresh",
        client_id="id",
        client_secret="secret"
    )
    assert result is True
    mock_dbx.files_upload.assert_called_once()

@mock.patch("src.upload_to_dropbox.refresh_access_token", return_value="mock_token")
@mock.patch("dropbox.Dropbox")
def test_upload_file_failure(mock_dbx_class, mock_refresh, temp_file):
    mock_dbx = mock.Mock()
    mock_dbx.files_upload.side_effect = Exception("Upload failed")
    mock_dbx_class.return_value = mock_dbx

    result = upload_file_to_dropbox(
        local_file_path=temp_file,
        dropbox_file_path="/mock/test_output.json",
        refresh_token="refresh",
        client_id="id",
        client_secret="secret"
    )
    assert result is False

def test_upload_file_missing_local_file():
    result = upload_file_to_dropbox(
        local_file_path="nonexistent.json",
        dropbox_file_path="/mock/nonexistent.json",
        refresh_token="refresh",
        client_id="id",
        client_secret="secret"
    )
    assert result is False



