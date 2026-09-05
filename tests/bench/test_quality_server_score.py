"""Logit → scalar-score mapping for the quality server.

The scoring service must support both head shapes we deploy:

- the FinePDFs regression head (1 logit, raw value clamped to [0, 3]);
- the fine-tuned ModernBERT classifier head (4 ordinal class logits,
  scored as the softmax expectation over class indices, continuously
  in [0, 3] so ``parquet.quality_threshold`` keeps working).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

# The import has to follow importorskip: at module level it
# would make collection fail on a box without torch instead of skipping.
from pdfsys_bench._quality_server import _logits_to_score  # noqa: E402


def test_single_logit_regression_passes_through():
    assert _logits_to_score(torch.tensor([[1.71]])) == pytest.approx(1.71)


def test_single_logit_regression_clamps_to_0_3():
    assert _logits_to_score(torch.tensor([[-0.5]])) == 0.0
    assert _logits_to_score(torch.tensor([[3.9]])) == 3.0


def test_four_class_head_scores_softmax_expectation():
    # Certain mass on class 3 → score 3.0; uniform logits → mean class 1.5.
    confident = torch.tensor([[-20.0, -20.0, -20.0, 20.0]])
    assert _logits_to_score(confident) == pytest.approx(3.0, abs=1e-4)

    uniform = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    assert _logits_to_score(uniform) == pytest.approx(1.5)


def test_four_class_head_interpolates_between_classes():
    # Equal mass on classes 1 and 2 → expectation 1.5.
    split = torch.tensor([[-20.0, 5.0, 5.0, -20.0]])
    assert _logits_to_score(split) == pytest.approx(1.5, abs=1e-4)
