"""Idempotent helper that ensures ~/magic-pdf.json sets the expected device-mode.

magic-pdf (MinerU) reads ~/magic-pdf.json on import to pick its device.
This module writes / updates that file before the VLM parser is first
loaded, so the runner can guarantee MPS (or CPU / CUDA) without asking
the user to hand-edit a JSON file.

Idempotency:
- file missing                  → write a fresh default + chosen device-mode
- file present, mode matches    → no-op
- file present, mode differs    → patch only the `device-mode` key, preserve
                                  every other key the user may have set
"""

from __future__ import annotations

import json
from pathlib import Path

# NOTE: does not honour MINERU_TOOLS_CONFIG_JSON env var — extend if CI ever sets it.
CONFIG_PATH = Path.home() / "magic-pdf.json"

_DEFAULT_MODELS_DIR = Path.home() / ".cache" / "mineru" / "models"


def ensure_config(device_mode: str) -> Path:
    """Make sure ~/magic-pdf.json has the requested device-mode.

    Parameters
    ----------
    device_mode:
        One of ``"cpu"``, ``"mps"``, ``"cuda"``. Other values are passed
        through verbatim — magic-pdf will reject unknowns at load time.

    Returns
    -------
    Path
        Absolute path to the (now-correct) config file.
    """
    if CONFIG_PATH.exists():
        try:
            existing = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
        if not isinstance(existing, dict):
            existing = {}
        if existing.get("device-mode") == device_mode:
            return CONFIG_PATH
        existing["device-mode"] = device_mode
        CONFIG_PATH.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return CONFIG_PATH

    _DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    default = {
        "device-mode": device_mode,
        "models-dir": str(_DEFAULT_MODELS_DIR),
        "table-config": {"model": "rapid_table", "enable": True},
        "formula-config": {"enable": True},
    }
    CONFIG_PATH.write_text(
        json.dumps(default, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return CONFIG_PATH
