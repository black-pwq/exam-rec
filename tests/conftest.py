import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run tests that initialize real OCR models",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-integration"):
        return

    skip = pytest.mark.skip(reason="requires --run-integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
