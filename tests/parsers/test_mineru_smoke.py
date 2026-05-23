"""Locks the mineru import surface used by both parser packages.

If these imports break, parsers can't work. Fast (no model load).
"""

from __future__ import annotations


def test_mineru_do_parse_importable() -> None:
    """Guards the primary mineru entry point; all parsers call do_parse()."""
    from mineru.cli.common import do_parse, prepare_env
    assert callable(do_parse)
    assert callable(prepare_env)


def test_mineru_backend_modules_importable() -> None:
    """Both pipeline and vlm backend modules import without cv2 errors,
    AND their internal entry points are callable.

    Importing these modules transitively loads cv2 — if the opencv-python/
    opencv-python-headless conflict resurfaces, this fails here.
    ``test_cv2_module_body_loaded`` guards the subtler case where cv2
    imports but is missing compiled constants.

    The deeper symbol asserts (``doc_analyze_streaming``, ``ModelSingleton``)
    lock the contract for downstream parser tasks — a mineru rename of
    these entry points must fail a CI check, not silently break parsers.
    """
    from mineru.backend.pipeline.pipeline_analyze import doc_analyze_streaming
    from mineru.backend.vlm.vlm_analyze import ModelSingleton
    assert callable(doc_analyze_streaming)
    assert callable(ModelSingleton)


def test_cv2_module_body_loaded() -> None:
    """Guard against opencv-python / opencv-python-headless conflict."""
    import cv2
    assert cv2.__file__ is not None
    assert hasattr(cv2, "INTER_NEAREST")
