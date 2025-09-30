import pytest
import os
from unittest.mock import patch, MagicMock
from src.download_dropbox_files import dropbox

# Mock the dropbox API client to avoid real network calls
@pytest.fixture
def mock_dropbox_client():
    with patch('dropbox.Dropbox', autospec=True) as mock_client:
        mock_client.return_value.files_list_folder.return_value.entries = [
            dropbox.files.FileMetadata(name='test_file.step', path_display='/test_file.step'),
            dropbox.files.FolderMetadata(name='test_folder')
        ]
        mock_client.return_value.files_download_to_file.return_value = (
            dropbox.files.FileMetadata(name='test_file.step'), None
        )
        yield mock_client.return_value

def test_download_all_files_successful(mock_dropbox_client, tmp_path):
    """Test that files are downloaded successfully."""
    download_all_files(mock_dropbox_client, '/test_folder', str(tmp_path))
    mock_dropbox_client.files_list_folder.assert_called_with('/test_folder')
    mock_dropbox_client.files_download_to_file.assert_called_with(os.path.join(tmp_path, 'test_file.step'), '/test_file.step')

def test_download_all_files_no_files(mock_dropbox_client, tmp_path):
    """Test behavior when the folder is empty."""
    mock_dropbox_client.files_list_folder.return_value.entries = []
    download_all_files(mock_dropbox_client, '/empty_folder', str(tmp_path))
    mock_dropbox_client.files_download_to_file.assert_not_called()

def test_download_all_files_api_error(mock_dropbox_client, tmp_path):
    """Test that a Dropbox API error is handled gracefully."""
    mock_dropbox_client.files_list_folder.side_effect = dropbox.exceptions.ApiError(
        request_id='mock_request_id',
        error=MagicMock(),
        user_message='mock_user_message'
    )
    with pytest.raises(dropbox.exceptions.ApiError):
        download_all_files(mock_dropbox_client, '/error_folder', str(tmp_path))
