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
    assert c.table_enable is False
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
    # Sibling fields keep their defaults — engine override must not mutate them.
    assert c.formula_enable is True
    assert c.table_enable is True
    assert c.p_lang == "ch"
    assert c.output_dir is None


def test_pipeline_config_serde_roundtrip() -> None:
    """Path | None must survive to_dict / from_dict round-trip."""
    from pdfsys_core.serde import from_dict, to_dict

    original = PipelineConfig(
        formula_enable=False,
        table_enable=True,
        p_lang="en",
        output_dir=Path("/tmp/example"),
    )
    blob = to_dict(original)
    restored = from_dict(PipelineConfig, blob)
    assert restored.formula_enable is False
    assert restored.p_lang == "en"
    assert restored.output_dir == Path("/tmp/example")


def test_vlm_config_serde_roundtrip_with_none_output() -> None:
    """output_dir=None must serialize to None (not 'None' string)."""
    from pdfsys_core.serde import from_dict, to_dict

    original = VlmConfig(engine="mlx-engine")
    blob = to_dict(original)
    assert blob["output_dir"] is None
    restored = from_dict(VlmConfig, blob)
    assert restored.engine == "mlx-engine"
    assert restored.output_dir is None
