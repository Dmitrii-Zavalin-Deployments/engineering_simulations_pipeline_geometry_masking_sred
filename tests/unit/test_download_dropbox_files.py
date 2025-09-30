import pytest
import os
import dropbox
from unittest.mock import MagicMock, patch
from src.download_dropbox_files import download_files_from_dropbox, refresh_access_token

# Mock objects for Dropbox API responses
class MockFileMetadata:
    def __init__(self, name, path_lower):
        self.name = name
        self.path_lower = path_lower

class MockResult:
    def __init__(self, entries, has_more=False, cursor=None):
        self.entries = entries
        self.has_more = has_more
        self.cursor = cursor

class MockDownloadResponse:
    def __init__(self, content):
        self.content = content

@pytest.fixture
def mock_dbx():
    """Fixture to mock the Dropbox object."""
    with patch('src.download_dropbox_files.dropbox.Dropbox') as mock_dbx_class:
        mock_instance = MagicMock()
        mock_dbx_class.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_refresh_token_success():
    """Fixture to mock a successful token refresh."""
    with patch('src.download_dropbox_files.refresh_access_token', return_value='fake_access_token') as mock:
        yield mock

@pytest.fixture
def mock_os_makedirs():
    """Fixture to mock os.makedirs to prevent folder creation."""
    with patch('src.download_dropbox_files.os.makedirs') as mock:
        yield mock

@pytest.fixture
def temp_log_file(tmp_path):
    """Fixture to create a temporary log file for testing."""
    log_file_path = tmp_path / "test.log"
    return str(log_file_path)

def test_refresh_access_token_success():
    """Test successful access token refresh."""
    with patch('src.download_dropbox_files.requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"access_token": "new_access_token"}
        mock_post.return_value = mock_response

        token = refresh_access_token("refresh_token", "client_id", "client_secret")
        assert token == "new_access_token"
        mock_post.assert_called_once()

def test_refresh_access_token_failure():
    """Test failed access token refresh."""
    with patch('src.download_dropbox_files.requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response

        with pytest.raises(Exception, match="Failed to refresh access token"):
            refresh_access_token("refresh_token", "client_id", "client_secret")

def test_download_files_from_dropbox_success(mock_dbx, mock_refresh_token_success, tmp_path, temp_log_file):
    """Test successful download of allowed files."""
    dropbox_folder = "/test_folder"
    local_folder = str(tmp_path / "local_downloads")

    # Mock Dropbox file list
    file1 = MockFileMetadata("test1.step", "/test_folder/test1.step")
    file2 = MockFileMetadata("test2.json", "/test_folder/test2.json")
    unsupported_file = MockFileMetadata("image.jpg", "/test_folder/image.jpg")

    mock_dbx.files_list_folder.return_value = MockResult(entries=[file1, file2, unsupported_file])
    mock_dbx.files_download.side_effect = [
        (None, MockDownloadResponse(b'content of step file')),
        (None, MockDownloadResponse(b'content of json file'))
    ]

    download_files_from_dropbox(dropbox_folder, local_folder, 'token', 'id', 'secret', temp_log_file)

    assert mock_dbx.files_list_folder.called
    assert mock_dbx.files_download.call_count == 2
    assert os.path.exists(os.path.join(local_folder, "test1.step"))
    assert os.path.exists(os.path.join(local_folder, "test2.json"))
    assert not os.path.exists(os.path.join(local_folder, "image.jpg"))

def test_download_files_from_dropbox_empty_folder(mock_dbx, mock_refresh_token_success, tmp_path, temp_log_file):
    """Test scenario with an empty Dropbox folder."""
    dropbox_folder = "/empty_folder"
    local_folder = str(tmp_path / "local_downloads")

    mock_dbx.files_list_folder.return_value = MockResult(entries=[])

    download_files_from_dropbox(dropbox_folder, local_folder, 'token', 'id', 'secret', temp_log_file)
    
    assert mock_dbx.files_list_folder.called
    assert mock_dbx.files_download.call_count == 0
    assert len(os.listdir(local_folder)) == 0

def test_download_files_from_dropbox_api_error(mock_dbx, mock_refresh_token_success, tmp_path, temp_log_file):
    """Test handling of Dropbox API errors."""
    dropbox_folder = "/error_folder"
    local_folder = str(tmp_path / "local_downloads")

    mock_dbx.files_list_folder.side_effect = dropbox.exceptions.ApiError(
        'error', 'error', 'error', 'error'
    )

    with patch('sys.stdout', new=MagicMock()) as mock_stdout: # Mock print statements
        download_files_from_dropbox(dropbox_folder, local_folder, 'token', 'id', 'secret', temp_log_file)
    
    assert "❌ Dropbox API error" in open(temp_log_file).read()

def test_download_files_from_dropbox_pagination(mock_dbx, mock_refresh_token_success, tmp_path, temp_log_file):
    """Test handling of paginated results from Dropbox API."""
    dropbox_folder = "/paginated_folder"
    local_folder = str(tmp_path / "local_downloads")

    file1 = MockFileMetadata("paginated1.step", "/paginated_folder/paginated1.step")
    file2 = MockFileMetadata("paginated2.stp", "/paginated_folder/paginated2.stp")

    # Mock first and second page of results
    mock_dbx.files_list_folder.return_value = MockResult(entries=[file1], has_more=True, cursor="cursor1")
    mock_dbx.files_list_folder_continue.return_value = MockResult(entries=[file2], has_more=False)
    
    # Mock download responses
    mock_dbx.files_download.side_effect = [
        (None, MockDownloadResponse(b'content of first file')),
        (None, MockDownloadResponse(b'content of second file'))
    ]
    
    download_files_from_dropbox(dropbox_folder, local_folder, 'token', 'id', 'secret', temp_log_file)
    
    assert mock_dbx.files_list_folder.called_once()
    assert mock_dbx.files_list_folder_continue.called_once()
    assert mock_dbx.files_download.call_count == 2
    assert os.path.exists(os.path.join(local_folder, "paginated1.step"))
    assert os.path.exists(os.path.join(local_folder, "paginated2.stp"))

def test_download_files_from_dropbox_unsupported_extensions(mock_dbx, mock_refresh_token_success, tmp_path, temp_log_file):
    """Test that files with unsupported extensions are skipped."""
    dropbox_folder = "/unsupported_folder"
    local_folder = str(tmp_path / "local_downloads")

    unsupported_file = MockFileMetadata("image.png", "/unsupported_folder/image.png")
    mock_dbx.files_list_folder.return_value = MockResult(entries=[unsupported_file])

    download_files_from_dropbox(dropbox_folder, local_folder, 'token', 'id', 'secret', temp_log_file)
    
    assert mock_dbx.files_list_folder.called
    assert mock_dbx.files_download.call_count == 0
    assert len(os.listdir(local_folder)) == 0
    assert "⏭️ Skipped file (unsupported type): image.png" in open(temp_log_file).read()

def test_download_files_from_dropbox_folder_not_found(mock_dbx, mock_refresh_token_success, tmp_path, temp_log_file):
    """Test handling of a non-existent Dropbox folder."""
    dropbox_folder = "/nonexistent_folder"
    local_folder = str(tmp_path / "local_downloads")

    mock_dbx.files_list_folder.side_effect = dropbox.exceptions.ApiError(
        'error', 'error', 'path', dropbox.files.ListFolderError.path_not_found('...'))
    
    with patch('sys.stdout', new=MagicMock()) as mock_stdout: # Mock print statements
        download_files_from_dropbox(dropbox_folder, local_folder, 'token', 'id', 'secret', temp_log_file)
    
    assert "❌ Dropbox API error" in open(temp_log_file).read()
    assert not os.path.exists(local_folder)



