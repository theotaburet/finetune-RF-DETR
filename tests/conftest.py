"""Pytest configuration for custom ini options."""


def pytest_addoption(parser) -> None:
    """Register custom ini options for real-file tests."""
    parser.addini(
        "real_audio_path",
        "Path to a real audio file used by real_files tests",
        default="",
    )
    parser.addini(
        "real_json_path",
        "Path to a real JSON metadata file used by real_files tests",
        default="",
    )
