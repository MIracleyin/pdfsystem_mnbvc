"""macOS workaround for mineru's PDF-render multiprocessing deadlock.

Mineru's ``mineru.utils.pdf_image_tools._get_pdf_render_executor`` returns
a ``ProcessPoolExecutor`` whose workers must re-import the parent
process's modules on ``spawn``. When the parent has loaded heavy state
(torch, MLX, transformers, our parsers), the spawn workers deadlock
during re-import.

Replacing the singleton with a ``ThreadPoolExecutor`` sidesteps the
issue because PyMuPDF / pypdfium2 are C extensions that release the
GIL during rendering — thread parallelism works without process
spawn.

This is darwin-only. On Linux with CUDA, mineru's mp.Pool design is
fine and faster than threads.
"""

from __future__ import annotations

import sys


def _install() -> None:
    if sys.platform != "darwin":
        return
    from concurrent.futures import ThreadPoolExecutor

    import mineru.utils.pdf_image_tools as _r

    # Singleton, lazy: created on first call so we don't spawn threads
    # we never use.
    _state: dict = {"executor": None}

    def _patched():
        if _state["executor"] is None:
            _state["executor"] = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="mineru-pdfrender",
            )
        return _state["executor"]

    _r._get_pdf_render_executor = _patched


_install()
