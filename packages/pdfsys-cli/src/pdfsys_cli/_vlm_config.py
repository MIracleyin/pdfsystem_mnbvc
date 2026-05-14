"""VLM engine environment setup for mineru 3.x.

mineru reads its device-mode and model-source from environment variables
(see ``out/mineru_spike_notes.md`` for the verified list). This module
sets those env vars before the VLM parser is first imported, and
relocates any legacy ~/magic-pdf.json out of the way so the old engine's
config can't surprise anyone.

Idempotency:
- env vars: set every call (env is session-local; cheap to overwrite).
- ~/magic-pdf.json: renamed to ~/magic-pdf.json.bak on first call, only
  if the source exists and the .bak target does not.
"""

from __future__ import annotations

import os
from pathlib import Path

# Verified in the migration spike (mineru 3.1.12 source).
ENV_DEVICE_MODE = "MINERU_DEVICE_MODE"
ENV_MODEL_SOURCE = "MINERU_MODEL_SOURCE"

LEGACY_CONFIG_PATH = Path.home() / "magic-pdf.json"
LEGACY_BACKUP_PATH = Path.home() / "magic-pdf.json.bak"


def ensure_mineru_env(device_mode: str, model_source: str = "huggingface") -> None:
    """Configure mineru's environment.

    Parameters
    ----------
    device_mode:
        One of ``"cpu"``, ``"mps"``, ``"cuda"``. Forwarded to mineru as
        ``MINERU_DEVICE_MODE``. Other values pass through verbatim.
    model_source:
        ``"huggingface"`` (default, fastest on most networks) or
        ``"modelscope"`` (China-friendly but throttled on large files
        per the spike on 2026-05-14). Forwarded as
        ``MINERU_MODEL_SOURCE``.
    """
    os.environ[ENV_DEVICE_MODE] = device_mode
    os.environ[ENV_MODEL_SOURCE] = model_source

    if LEGACY_CONFIG_PATH.exists() and not LEGACY_BACKUP_PATH.exists():
        LEGACY_CONFIG_PATH.rename(LEGACY_BACKUP_PATH)
