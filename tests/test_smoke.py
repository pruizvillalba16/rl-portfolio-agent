"""Smoke test: the installed package can be imported."""

import rl_portfolio


def test_package_imports() -> None:
    assert rl_portfolio.__version__ == "0.1.0"
