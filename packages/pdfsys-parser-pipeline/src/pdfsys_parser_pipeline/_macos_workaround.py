"""macOS workaround for mineru's PDF-render multiprocessing deadlock.

Root cause: mineru spawns subprocesses for PDF rendering and model init.
On macOS the default start method is ``spawn``, which forces children to
re-import the parent's modules. When the parent has loaded a heavy
import surface (torch + MLX + transformers + our parsers), the spawn
children deadlock during re-import.

Fix: switch to ``fork`` so children inherit the parent's loaded state
via copy-on-write. macOS's Objective-C runtime warns about fork from
multi-threaded processes — ``OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES``
suppresses that and lets fork proceed (we don't touch CoreFoundation
across the fork boundary).

Verified: mineru pipeline mode 25.4s/page, mineru VLM mlx-engine
14s/page after this fix.

Linux + CUDA is unaffected — the original spawn-mode code works there.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import sys


def _install() -> None:
    if sys.platform != "darwin":
        return

    # Must be set before any mp.Pool / ProcessPoolExecutor spawn workers.
    # Setting before import-time is best; setting later with force=True is
    # also fine because mp's start_method is a process-global one-shot.
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

    # Already-set is acceptable as long as it's "fork".
    with contextlib.suppress(RuntimeError):
        mp.set_start_method("fork", force=True)


_install()
