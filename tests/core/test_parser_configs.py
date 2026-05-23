"""Pin the new shape of PipelineConfig + VlmConfig (mineru-flavored)."""

from __future__ import annotations

from pathlib import Path

from pdfsys_core import PipelineConfig, VlmConfig


def test_pipeline_config_defaults() -> None:
    c = PipelineConfig()
    assert c.formula_enable is True
    assert c.table_enable is True
    assert c.p_lang == "ch"
    assert c.output_dir is None


def test_pipeline_config_override() -> None:
    c = PipelineConfig(
        formula_enable=False, table_enable=False, p_lang="en",
        output_dir=Path("/tmp/x"),
    )
    assert c.formula_enable is False
    assert c.p_lang == "en"
    assert c.output_dir == Path("/tmp/x")


def test_vlm_config_defaults() -> None:
    c = VlmConfig()
    assert c.engine == "transformers"
    assert c.formula_enable is True
    assert c.p_lang == "ch"
    assert c.output_dir is None


def test_vlm_config_engine_override() -> None:
    c = VlmConfig(engine="mlx-engine")
    assert c.engine == "mlx-engine"
