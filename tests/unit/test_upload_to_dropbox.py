import pytest
from unittest.mock import patch, MagicMock
from src.upload_to_dropbox import upload_file_to_dropbox, dropbox

# Mock the dropbox API client
@pytest.fixture
def mock_dropbox_client():
    with patch('dropbox.Dropbox', autospec=True) as mock_client:
        mock_client.return_value.files_upload.return_value = dropbox.files.FileMetadata(name='uploaded.json')
        yield mock_client.return_value

def test_upload_file_successful(mock_dropbox_client, tmp_path):
    """Test that a file is uploaded successfully."""
    # Create a dummy local file to upload
    local_file_path = tmp_path / "test_upload.json"
    local_file_path.write_text("dummy content")

    upload_file(mock_dropbox_client, str(local_file_path), '/remote_path/test_upload.json')
    
    # Assert that the upload method was called with the correct arguments
    mock_dropbox_client.files_upload.assert_called_once()
    assert mock_dropbox_client.files_upload.call_args[0][1] == '/remote_path/test_upload.json'

def test_upload_file_does_not_exist(mock_dropbox_client):
    """Test behavior when the local file does not exist."""
    with pytest.raises(FileNotFoundError):
        upload_file(mock_dropbox_client, 'non_existent_file.json', '/remote_path/file.json')

def test_upload_file_api_error(mock_dropbox_client, tmp_path):
    """Test that a Dropbox API error is handled gracefully."""
    local_file_path = tmp_path / "test_error_upload.json"
    local_file_path.write_text("dummy content")
    mock_dropbox_client.files_upload.side_effect = dropbox.exceptions.ApiError(
        request_id='mock_request_id',
        error=MagicMock(),
        user_message='mock_user_message'
    )
    with pytest.raises(dropbox.exceptions.ApiError):
        upload_file(mock_dropbox_client, str(local_file_path), '/remote_path/error.json')
