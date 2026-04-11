"""Thin loader around the FinePDFs XGBoost ``xgb.ubj`` weights.

The model is a binary classifier where class 1 = "needs OCR" (scanned /
garbled / image-heavy / form). It takes a 124-column feature vector whose
column order is fixed by :func:`feature_extractor.flatten_per_page_features`.

We keep the loader tiny on purpose: the calibration between feature layout
and column order lives entirely in ``feature_extractor.py`` — this file
only knows "give me a dict-of-features, I'll give you a probability".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


class XgbRouterModel:
    """Lazy-loading wrapper around an ``xgb.ubj`` binary classifier."""

    def __init__(self, path_to_model: str | Path) -> None:
        self.path_to_model = Path(path_to_model)
        self._model: XGBClassifier | None = None

    @property
    def model(self) -> XGBClassifier:
        if self._model is None:
            if not self.path_to_model.is_file():
                raise FileNotFoundError(
                    f"XGBoost weights not found at {self.path_to_model}. "
                    "Run `python -m pdfsys_router.download_weights` to fetch them."
                )
            m = XGBClassifier()
            m.load_model(str(self.path_to_model))
            self._model = m
        return self._model

    def predict_proba(self, features: dict[str, float]) -> float:
        """Return P(class=1, i.e. needs OCR)."""
        df = pd.DataFrame([features])
        # Column ordering must match the training schema — realign using
        # the model's recorded feature_names_in_ when available.
        names = getattr(self.model, "feature_names_in_", None)
        if names is not None:
            df = df.reindex(columns=list(names), fill_value=0)
        probs = self.model.predict_proba(df)
        return float(probs[0][1])

    @property
    def feature_names(self) -> list[str]:
        names = getattr(self.model, "feature_names_in_", None)
        if names is None:
            return []
        return list(names)

    @property
    def n_features(self) -> int:
        return int(getattr(self.model, "n_features_in_", 0))


def default_weights_path() -> Path:
    """Return the canonical on-disk location of the bundled weights."""
    return Path(__file__).resolve().parent.parent.parent / "models" / "xgb_classifier.ubj"
