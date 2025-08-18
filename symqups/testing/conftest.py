import pytest

def pytest_addoption(parser):
    parser.addoption("--full", action="store_true", help="Run full test suite")

def pytest_configure(config):
    config.addinivalue_line("markers", "full: mark test as full/slow")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--full"):
        # run everything
        return
    skip_full = pytest.mark.skip(reason="need --full option to run")
    for item in items:
        if "full" in item.keywords:
            item.add_marker(skip_full)