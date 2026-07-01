from openptv2 import validate as runtime_validate


def test_runtime_suite_without_legacy_requirement():
    results = runtime_validate.run_validation_suite(
        tolerance=1e-10,
        benchmark=False,
        iterations=1,
        min_speed_ratio=None,
        require_legacy_baseline=False,
    )

    assert results
    assert all(result.status in {"PASS", "SKIP"} for result in results)
    assert any(result.name == "runtime_info" for result in results)


def test_runtime_suite_requires_legacy_baseline_when_requested():
    results = runtime_validate.run_validation_suite(
        tolerance=1e-10,
        benchmark=False,
        iterations=1,
        min_speed_ratio=None,
        require_legacy_baseline=True,
    )

    if runtime_validate._legacy_modules() is None:
        assert results[-1].name == "legacy_baseline"
        assert results[-1].status == "FAIL"
    else:
        assert all(result.status in {"PASS", "SKIP"} for result in results)
