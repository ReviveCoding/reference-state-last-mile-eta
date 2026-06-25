import pytest

from scripts.run_pipeline import _validated_output_namespace


def test_output_namespace_rejects_path_traversal() -> None:
    assert _validated_output_namespace("gpu_smoke") == "gpu_smoke"
    with pytest.raises(ValueError, match="output_namespace"):
        _validated_output_namespace("../../escape")
    with pytest.raises(ValueError, match="output_namespace"):
        _validated_output_namespace("bad/name")
