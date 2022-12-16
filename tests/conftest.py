import pytest


@pytest.fixture
def hello_world() -> str:
    return "hello world!"
