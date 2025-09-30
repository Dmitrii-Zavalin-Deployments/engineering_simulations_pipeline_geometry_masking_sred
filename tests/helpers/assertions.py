def assert_error_contains(exception, substring):
    """Asserts that an exception's message contains a specific substring."""
    assert substring in str(exception.value)
