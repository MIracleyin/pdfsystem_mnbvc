"""Tests for stage resolution in :mod:`pdfsys_cli.config`.

The specific failure being guarded: ``--no-quality`` also removes ``parquet``
(the L1 ``kept`` column is decided by ``quality_score``), and it used to do so
without a word. Asking for a stage and silently not getting it is the kind of
thing you discover hours later, looking for output that was never written.
"""

from __future__ import annotations

from pdfsys_cli.config import apply_cli_overrides, default_config


def test_no_quality_records_the_stages_it_dropped():
    cfg = apply_cli_overrides(default_config(), no_quality=True)

    assert "quality" not in cfg.stages
    assert "parquet" not in cfg.stages
    assert cfg.dropped_stages == ["quality", "parquet"]


def test_a_run_that_dropped_nothing_reports_nothing():
    cfg = apply_cli_overrides(default_config(), stages="router,extract")

    assert cfg.dropped_stages == []


def test_only_stages_that_were_actually_requested_are_reported():
    """No phantom warning about removing a stage the user never asked for."""
    cfg = apply_cli_overrides(
        default_config(), stages="router,extract", no_quality=True
    )

    assert cfg.stages == ["router", "extract"]
    assert cfg.dropped_stages == []


def test_resume_drops_parquet_because_parquet_cannot_be_appended_to():
    """pq.ParquetWriter truncates on open, so a resumed leg would leave
    dataset.parquet describing only that leg while results.jsonl describes the
    whole run — two artifacts of one run silently disagreeing."""
    cfg = apply_cli_overrides(default_config(), resume=True)

    assert "parquet" not in cfg.stages
    assert cfg.dropped_stages == ["parquet"]
    assert "not appendable" in cfg.drop_reasons["parquet"]


def test_a_resumed_run_that_never_wanted_parquet_says_nothing():
    cfg = apply_cli_overrides(
        default_config(), stages="router,extract", resume=True
    )

    assert cfg.dropped_stages == []


def test_every_dropped_stage_carries_its_reason():
    cfg = apply_cli_overrides(default_config(), no_quality=True, resume=True)

    assert set(cfg.drop_reasons) == set(cfg.dropped_stages)
    assert all(cfg.drop_reasons[s] for s in cfg.dropped_stages)


def test_asking_for_parquet_and_then_disabling_quality_reports_both():
    """``parquet`` auto-includes ``quality`` as a dependency, so ``--no-quality``
    takes out a stage that was never typed. That is exactly the combination
    worth naming out loud."""
    cfg = apply_cli_overrides(
        default_config(), stages="router,extract,parquet", no_quality=True
    )

    assert cfg.stages == ["router", "extract"]
    assert cfg.dropped_stages == ["quality", "parquet"]
