"""Unit tests for openptv2 self-check diagnostic runner."""

from openptv2.self_check import SelfChecker, main as self_check_main


def test_self_check_runner():
    checker = SelfChecker(verbose=True, strict=False)
    checks = checker.run_all()

    assert checks
    categories = {c.category for c in checks}
    assert "Environment" in categories
    assert "Dependencies" in categories
    assert "Cython Runtime" in categories
    assert "Core API" in categories
    assert "Validation Suite" in categories

    # Environment checks should pass
    env_checks = [c for c in checks if c.category == "Environment"]
    assert all(c.status == "PASS" for c in env_checks)

    # Core API checks should pass
    api_checks = [c for c in checks if c.category == "Core API"]
    assert all(c.status == "PASS" for c in api_checks)


def test_self_check_main_entry_point():
    ret = self_check_main(["--quiet"])
    assert ret == 0
