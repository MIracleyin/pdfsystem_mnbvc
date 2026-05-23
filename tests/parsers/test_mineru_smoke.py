"""Locks the mineru import surface used by both parser packages.

If these imports break, parsers can't work. Fast (no model load).
"""

from __future__ import annotations


def test_mineru_do_parse_importable() -> None:
    from mineru.cli.common import do_parse, prepare_env
    assert callable(do_parse)
    assert callable(prepare_env)


def test_mineru_backend_modules_importable() -> None:
    """Both pipeline and vlm backend modules import without cv2 errors."""
    from mineru.backend.pipeline import pipeline_analyze  # noqa: F401
    from mineru.backend.vlm import vlm_analyze  # noqa: F401


def test_cv2_module_body_loaded() -> None:
    """Guard against opencv-python / opencv-python-headless conflict."""
    import cv2
    assert cv2.__file__ is not None
    assert hasattr(cv2, "INTER_NEAREST")
