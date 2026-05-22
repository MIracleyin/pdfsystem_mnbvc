# Release Gate Layer 4 + Calibration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a downstream `release_gate.py` that turns the existing bench JSONL into a per-PDF `release_manifest.jsonl` with `publish | review | reject` decisions, plus an offline `llm_review.py`, a TOML threshold profile, a calibration set scaffold, and a `fit_profile.py` for threshold search.

**Architecture:** Three new modules under `packages/pdfsys-bench/src/pdfsys_bench/` (`release_gate.py`, `llm_review.py`, `fit_profile.py`), all reading the existing bench JSONL. No changes to `loop.py` or `cascade.py`. LLM-judge stays strictly out of the decision rule. Calibration assets live under `packages/pdfsys-bench/calibration/`. Viz gets a `POST /api/label` endpoint + a labeling control on the detail card so v0 calibration can happen in-browser.

**Tech Stack:** Python 3.11+, stdlib `tomllib` for TOML, pytest for tests, ruff (select E,F,W,I,UP,B,SIM,RUF) for linting, stdlib `http.server` for the viz server. LLM access via the existing `pdfsys_bench.llm_client` wrapper.

**Source spec:** `docs/superpowers/specs/2026-05-22-release-gate-layer4-design.md`

**Conventions in this codebase (from reading existing code):**
- `from __future__ import annotations` at the top of every Python file.
- `@dataclass(slots=True)` for data containers.
- Module-private helpers named `_snake_case`.
- Module constants `DEFAULT_X = ...` at the top.
- Tests in `tests/bench/test_<module>.py`; use plain functions, no test classes.
- One commit per task; commit message format: `feat(bench): <one-line summary>`.
- Run a single test: `uv run pytest tests/bench/test_<module>.py::test_<name> -v`.
- Run the whole bench suite: `uv run pytest tests/bench/ -v`.
- Lint: `uv run ruff check packages/pdfsys-bench`.

---

## File Structure

**New files:**

```
packages/pdfsys-bench/src/pdfsys_bench/
├── release_gate.py            # decide() + load_profile() + run_gate() + __main__
├── llm_review.py              # offline LLM scoring with --llm-scope + resume
└── fit_profile.py             # threshold search over labels.jsonl → TOML

packages/pdfsys-bench/calibration/
├── README.md                  # calibration protocol + how to re-fit
├── profiles/
│   └── default-v1.toml        # initial profile with hand-set thresholds
└── labels.jsonl               # empty initially; populated by llm_review + humans

tests/bench/
├── test_release_gate.py       # decide(), load_profile(), run_gate()
├── test_llm_review.py         # scope filter, resume, mocked client
└── test_fit_profile.py        # threshold search on synthetic labels
```

**Modified files:**

- `packages/pdfsys-bench/viz/viz_server.py` — add `GET /api/labels`, `POST /api/label` endpoints (mirror the existing `_load_badcases` / `_append_badcase` pattern with a separate `LABELS_PATH`).
- `packages/pdfsys-bench/viz/index.html` — add a labeling control to the detail card.

**Unchanged (do not edit):** `loop.py`, `cascade.py`, `quality.py`, `quality_rules.py`, `quality_llm.py`, `llm_client.py`. The whole point of this design is that release-gate sits downstream of the loop.

---

## Task 1: TOML profile loader + default-v1.toml

Single source of truth for thresholds. Everything else reads `ThresholdProfile`.

**Files:**
- Create: `packages/pdfsys-bench/src/pdfsys_bench/release_gate.py`
- Create: `packages/pdfsys-bench/calibration/profiles/default-v1.toml`
- Create: `tests/bench/test_release_gate.py`

- [ ] **Step 1: Write failing tests for the loader**

Create `tests/bench/test_release_gate.py`:

```python
"""Release-gate decision engine tests (Layer 4)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from pdfsys_bench.release_gate import ThresholdProfile, load_profile


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "profile.toml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


def test_load_profile_happy_path(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        name = "test"
        version = "1.0.0"
        created_at = "2026-05-22"
        description = "fixture"

        [grade_boundaries]
        excellent = 2.5
        good = 1.5
        fair = 0.5

        [decision]
        t_publish = 2.0
        t_reject = 0.5

        [blockers]
        disable = []
    """)
    profile = load_profile(p)
    assert isinstance(profile, ThresholdProfile)
    assert profile.name == "test"
    assert profile.version == "1.0.0"
    assert profile.t_publish == 2.0
    assert profile.t_reject == 0.5
    assert profile.grade_boundaries == {"excellent": 2.5, "good": 1.5, "fair": 0.5}
    assert profile.disabled_blockers == frozenset()
    assert profile.identifier == "test@1.0.0"


def test_load_profile_rejects_missing_name(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        version = "1.0.0"
        created_at = "2026-05-22"
        description = ""
        [grade_boundaries]
        excellent = 2.5
        good = 1.5
        fair = 0.5
        [decision]
        t_publish = 2.0
        t_reject = 0.5
        [blockers]
        disable = []
    """)
    with pytest.raises(ValueError, match="name"):
        load_profile(p)


def test_load_profile_rejects_t_reject_above_t_publish(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        name = "bad"
        version = "1.0.0"
        created_at = "2026-05-22"
        description = ""
        [grade_boundaries]
        excellent = 2.5
        good = 1.5
        fair = 0.5
        [decision]
        t_publish = 1.0
        t_reject = 2.0
        [blockers]
        disable = []
    """)
    with pytest.raises(ValueError, match="t_reject.*<.*t_publish"):
        load_profile(p)


def test_load_profile_rejects_non_monotonic_grades(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        name = "bad"
        version = "1.0.0"
        created_at = "2026-05-22"
        description = ""
        [grade_boundaries]
        excellent = 1.5
        good = 2.5
        fair = 0.5
        [decision]
        t_publish = 2.0
        t_reject = 0.5
        [blockers]
        disable = []
    """)
    with pytest.raises(ValueError, match="grade.*monotonic"):
        load_profile(p)


def test_load_profile_parses_disabled_blockers(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        name = "test"
        version = "1.0.0"
        created_at = "2026-05-22"
        description = ""
        [grade_boundaries]
        excellent = 2.5
        good = 1.5
        fair = 0.5
        [decision]
        t_publish = 2.0
        t_reject = 0.5
        [blockers]
        disable = ["repetition_loop", "too_short"]
    """)
    profile = load_profile(p)
    assert profile.disabled_blockers == frozenset({"repetition_loop", "too_short"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/bench/test_release_gate.py -v`
Expected: 5 failures, all `ImportError: cannot import name 'ThresholdProfile' from 'pdfsys_bench.release_gate'`.

- [ ] **Step 3: Implement `release_gate.py` (loader portion only)**

Create `packages/pdfsys-bench/src/pdfsys_bench/release_gate.py`:

```python
"""Release-gate engine — turns bench JSONL into a release manifest.

This module sits DOWNSTREAM of the bench loop. It must not import from
``pdfsys_bench.loop`` or ``pdfsys_bench.cascade``; it consumes their
JSONL output only.

See ``docs/superpowers/specs/2026-05-22-release-gate-layer4-design.md``.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

SCORER_VERSION = "release-gate-v0.1"


@dataclass(slots=True, frozen=True)
class ThresholdProfile:
    """Decoded threshold profile. Loaded from a TOML file.

    The ``identifier`` (``<name>@<version>``) is written into every
    manifest row's ``threshold_profile`` field so downstream consumers
    can trace decisions back to a specific policy.
    """

    name: str
    version: str
    created_at: str
    description: str
    t_publish: float
    t_reject: float
    grade_boundaries: dict[str, float]
    disabled_blockers: frozenset[str] = field(default_factory=frozenset)

    @property
    def identifier(self) -> str:
        return f"{self.name}@{self.version}"


_REQUIRED_TOP_KEYS = ("name", "version", "created_at", "description")
_REQUIRED_GRADE_KEYS = ("excellent", "good", "fair")


def load_profile(path: str | Path) -> ThresholdProfile:
    """Parse a TOML profile and validate it.

    Raises:
        FileNotFoundError: if ``path`` doesn't exist.
        ValueError: on missing keys, non-monotonic grade boundaries, or
                    ``t_reject >= t_publish``.
    """
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)

    for key in _REQUIRED_TOP_KEYS:
        if key not in raw:
            raise ValueError(f"profile missing required key: {key!r}")

    grades_raw = raw.get("grade_boundaries", {})
    for key in _REQUIRED_GRADE_KEYS:
        if key not in grades_raw:
            raise ValueError(f"profile missing grade_boundaries.{key}")

    decision_raw = raw.get("decision", {})
    if "t_publish" not in decision_raw or "t_reject" not in decision_raw:
        raise ValueError("profile missing decision.t_publish or decision.t_reject")

    t_publish = float(decision_raw["t_publish"])
    t_reject = float(decision_raw["t_reject"])
    if not (t_reject < t_publish):
        raise ValueError(
            f"profile requires t_reject < t_publish; "
            f"got t_reject={t_reject}, t_publish={t_publish}"
        )

    grades = {k: float(grades_raw[k]) for k in _REQUIRED_GRADE_KEYS}
    # Strict descending: excellent > good > fair
    if not (grades["excellent"] > grades["good"] > grades["fair"]):
        raise ValueError(
            f"grade boundaries must be strictly monotonic descending; got {grades}"
        )

    blockers_raw = raw.get("blockers", {})
    disabled = frozenset(blockers_raw.get("disable", []))

    return ThresholdProfile(
        name=str(raw["name"]),
        version=str(raw["version"]),
        created_at=str(raw["created_at"]),
        description=str(raw["description"]),
        t_publish=t_publish,
        t_reject=t_reject,
        grade_boundaries=grades,
        disabled_blockers=disabled,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/bench/test_release_gate.py -v`
Expected: 5 passed.

- [ ] **Step 5: Create the initial profile file**

Create `packages/pdfsys-bench/calibration/profiles/default-v1.toml`:

```toml
# Release-gate threshold profile for pdfsys-bench.
# Owned by docs/superpowers/specs/2026-05-22-release-gate-layer4-design.md (§7).
# Thresholds here are HAND-SET starting points; re-fit with fit_profile.py
# once calibration/labels.jsonl has enough human labels and bump `version`.
name = "default-v1"
version = "0.1.0"
created_at = "2026-05-22"
description = "v0 calibration on OmniDocBench 150 samples — placeholder thresholds before fitting"

[grade_boundaries]
# Lower bounds, inclusive. score >= excellent => "excellent", etc.
excellent = 2.5
good = 1.5
fair = 0.5
# score < fair  => "poor".

[decision]
t_publish = 2.0   # score >= t_publish AND no blockers => publish
t_reject  = 0.5   # score <  t_reject  => reject (also any blocker => reject)
# t_reject <= score < t_publish => review

[blockers]
# All Layer-1 blockers count by default. Add a name here to ignore it
# in this profile. Use sparingly — prefer fixing the blocker upstream.
disable = []
```

- [ ] **Step 6: Verify the bundled profile loads**

Run: `uv run python -c "from pathlib import Path; from pdfsys_bench.release_gate import load_profile; p = load_profile(Path('packages/pdfsys-bench/calibration/profiles/default-v1.toml')); print(p.identifier, p.t_publish, p.t_reject)"`
Expected output: `default-v1@0.1.0 2.0 0.5`

- [ ] **Step 7: Lint**

Run: `uv run ruff check packages/pdfsys-bench/src/pdfsys_bench/release_gate.py`
Expected: `All checks passed!`

- [ ] **Step 8: Commit**

```bash
git add packages/pdfsys-bench/src/pdfsys_bench/release_gate.py \
        packages/pdfsys-bench/calibration/profiles/default-v1.toml \
        tests/bench/test_release_gate.py
git commit -m "feat(bench): release-gate TOML profile loader + default-v1 profile"
```

---

## Task 2: `decide()` pure logic

The decision function. Pure: takes a bench row dict + profile, returns `(decision, grade, reasons)`.

**Files:**
- Modify: `packages/pdfsys-bench/src/pdfsys_bench/release_gate.py`
- Modify: `tests/bench/test_release_gate.py`

- [ ] **Step 1: Append failing tests**

Add to `tests/bench/test_release_gate.py`:

```python
from pdfsys_bench.release_gate import decide, grade_for_score


def _profile(t_publish: float = 2.0, t_reject: float = 0.5) -> ThresholdProfile:
    return ThresholdProfile(
        name="t", version="1.0.0", created_at="2026-05-22", description="",
        t_publish=t_publish, t_reject=t_reject,
        grade_boundaries={"excellent": 2.5, "good": 1.5, "fair": 0.5},
        disabled_blockers=frozenset(),
    )


def _bench_row(
    *,
    quality_score: float | None = 2.3,
    cascade_attempts: list[dict] | None = None,
) -> dict:
    return {
        "sha256": "abc123",
        "quality_score": quality_score,
        "cascade_decision": "publish",
        "cascade_final_stage": "mupdf",
        "cascade_attempts": cascade_attempts
            or [{"stage": "mupdf", "decision": "publish",
                 "blockers": {"empty_output": False, "too_short": False,
                              "high_replacement_chars": False, "high_garbage_chars": False,
                              "repetition_loop": False},
                 "metrics": {}, "error": None, "wall_ms": 1.0}],
    }


def test_decide_publish_when_score_above_t_publish_and_no_blockers() -> None:
    decision, grade, reasons = decide(_bench_row(quality_score=2.3), _profile())
    assert decision == "publish"
    assert grade == "good"
    assert any("t_publish" in r for r in reasons)


def test_decide_reject_when_blocker_triggered() -> None:
    bad = _bench_row(
        quality_score=2.8,  # high score, but blocker vetoes
        cascade_attempts=[{
            "stage": "vlm",
            "decision": "escalate",
            "blockers": {"empty_output": False, "too_short": False,
                         "high_replacement_chars": False, "high_garbage_chars": False,
                         "repetition_loop": True},
            "metrics": {},
            "error": None,
            "wall_ms": 1.0,
        }],
    )
    decision, grade, reasons = decide(bad, _profile())
    assert decision == "reject"
    assert any("repetition_loop" in r for r in reasons)


def test_decide_reject_when_score_below_t_reject() -> None:
    decision, _, reasons = decide(_bench_row(quality_score=0.2), _profile())
    assert decision == "reject"
    assert any("t_reject" in r for r in reasons)


def test_decide_review_in_grey_band() -> None:
    decision, grade, reasons = decide(_bench_row(quality_score=1.0), _profile())
    assert decision == "review"
    assert grade == "fair"
    assert any("grey band" in r for r in reasons)


def test_decide_review_when_score_missing() -> None:
    decision, grade, reasons = decide(_bench_row(quality_score=None), _profile())
    assert decision == "review"
    assert grade is None
    assert any("missing" in r for r in reasons)


def test_decide_ignores_disabled_blockers() -> None:
    profile = ThresholdProfile(
        name="t", version="1.0.0", created_at="2026-05-22", description="",
        t_publish=2.0, t_reject=0.5,
        grade_boundaries={"excellent": 2.5, "good": 1.5, "fair": 0.5},
        disabled_blockers=frozenset({"repetition_loop"}),
    )
    bad = _bench_row(
        quality_score=2.5,
        cascade_attempts=[{
            "stage": "mupdf", "decision": "publish",
            "blockers": {"empty_output": False, "too_short": False,
                         "high_replacement_chars": False, "high_garbage_chars": False,
                         "repetition_loop": True},
            "metrics": {}, "error": None, "wall_ms": 1.0,
        }],
    )
    decision, _, _ = decide(bad, profile)
    assert decision == "publish"


def test_grade_for_score_thresholds() -> None:
    profile = _profile()
    assert grade_for_score(2.5, profile) == "excellent"
    assert grade_for_score(2.49, profile) == "good"
    assert grade_for_score(1.5, profile) == "good"
    assert grade_for_score(0.5, profile) == "fair"
    assert grade_for_score(0.49, profile) == "poor"
    assert grade_for_score(None, profile) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/bench/test_release_gate.py -v`
Expected: 7 new failures, all `ImportError: cannot import name 'decide'`.

- [ ] **Step 3: Add `decide()` and `grade_for_score()` to `release_gate.py`**

Append to `packages/pdfsys-bench/src/pdfsys_bench/release_gate.py`:

```python
DECISION_PUBLISH = "publish"
DECISION_REVIEW = "review"
DECISION_REJECT = "reject"


def grade_for_score(score: float | None, profile: ThresholdProfile) -> str | None:
    """Map a numeric score to a grade label via the profile boundaries.

    Returns ``None`` if ``score`` is ``None`` (e.g. the bench was run
    with ``--no-quality``).
    """
    if score is None:
        return None
    g = profile.grade_boundaries
    if score >= g["excellent"]:
        return "excellent"
    if score >= g["good"]:
        return "good"
    if score >= g["fair"]:
        return "fair"
    return "poor"


def _final_blockers(row: dict) -> dict[str, bool]:
    """Pull Layer-1 blockers off the last cascade attempt (if any).

    Non-cascade rows have no ``cascade_attempts``; treat them as empty.
    """
    attempts = row.get("cascade_attempts") or []
    if not attempts:
        return {}
    last = attempts[-1]
    return dict(last.get("blockers") or {})


def decide(
    row: dict,
    profile: ThresholdProfile,
) -> tuple[str, str | None, list[str]]:
    """Apply the release-gate rules to one bench row.

    Returns:
        ``(decision, grade, reasons)`` where ``decision`` is one of
        ``"publish"``, ``"review"``, ``"reject"``; ``grade`` is the
        score-derived label (or ``None`` when score is missing); and
        ``reasons`` is a human-readable trail.

    Order:
        1. Any non-disabled Layer-1 blocker → reject (vetoes score).
        2. score is None → review (with explanatory reason).
        3. score >= t_publish → publish.
        4. score < t_reject → reject.
        5. else → review (grey band).
    """
    reasons: list[str] = []
    grade = grade_for_score(row.get("quality_score"), profile)

    blockers = _final_blockers(row)
    triggered = [
        name
        for name, hit in blockers.items()
        if hit and name not in profile.disabled_blockers
    ]
    if triggered:
        reasons.append(f"Layer-1 blockers triggered: {triggered}")
        return DECISION_REJECT, grade, reasons

    score = row.get("quality_score")
    if score is None:
        reasons.append("doc_quality_score missing — routed to review")
        return DECISION_REVIEW, grade, reasons

    if score >= profile.t_publish:
        reasons.append(
            f"doc_quality_score={score:.2f} >= t_publish={profile.t_publish:.2f}"
        )
        reasons.append("no Layer-1 blockers triggered")
        return DECISION_PUBLISH, grade, reasons

    if score < profile.t_reject:
        reasons.append(
            f"doc_quality_score={score:.2f} < t_reject={profile.t_reject:.2f}"
        )
        return DECISION_REJECT, grade, reasons

    reasons.append(
        f"doc_quality_score={score:.2f} in grey band "
        f"[{profile.t_reject:.2f}, {profile.t_publish:.2f}) — needs review"
    )
    return DECISION_REVIEW, grade, reasons
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/bench/test_release_gate.py -v`
Expected: 12 passed (5 from Task 1 + 7 new).

- [ ] **Step 5: Lint**

Run: `uv run ruff check packages/pdfsys-bench/src/pdfsys_bench/release_gate.py tests/bench/test_release_gate.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add packages/pdfsys-bench/src/pdfsys_bench/release_gate.py tests/bench/test_release_gate.py
git commit -m "feat(bench): release-gate decide() — 3-state decision with blocker veto"
```

---

## Task 3: `run_gate()` end-to-end + `__main__` CLI

Wire JSONL-in / JSONL-out + CLI entry. Build manifest rows that match the §5 schema.

**Files:**
- Modify: `packages/pdfsys-bench/src/pdfsys_bench/release_gate.py`
- Modify: `tests/bench/test_release_gate.py`

- [ ] **Step 1: Append failing tests**

Add to `tests/bench/test_release_gate.py`:

```python
import json

from pdfsys_bench.release_gate import build_manifest_row, run_gate


def test_build_manifest_row_publish_with_cascade(tmp_path: Path) -> None:
    profile = _profile()
    row = _bench_row(quality_score=2.3)
    manifest = build_manifest_row(row, profile)
    assert manifest["doc_id"] == "abc123"
    assert manifest["decision"] == "publish"
    assert manifest["doc_quality_score"] == 2.3
    assert manifest["doc_quality_grade"] == "good"
    assert manifest["blockers"] == {
        "empty_output": False, "too_short": False,
        "high_replacement_chars": False, "high_garbage_chars": False,
        "repetition_loop": False,
    }
    assert isinstance(manifest["reasons"], list) and manifest["reasons"]
    assert manifest["cascade_final_stage"] == "mupdf"
    # v1 page-level + Layer-3 fields are reserved as null
    assert manifest["page_quality_p05"] is None
    assert manifest["page_quality_min"] is None
    assert manifest["bad_page_ratio"] is None
    assert manifest["visual_alignment_score"] is None
    assert manifest["consensus_score"] is None
    # Traceability fields
    assert manifest["scorer_version"] == "release-gate-v0.1"
    assert manifest["threshold_profile"] == "t@1.0.0"
    # LLM fields are null until llm_review runs
    assert manifest["quality_score_llm"] is None
    assert manifest["quality_reason_llm"] is None
    assert manifest["quality_model_llm"] is None


def test_build_manifest_row_non_cascade_run() -> None:
    """When the bench was run without --cascade, blockers must be {} and
    cascade_final_stage None — decide() then falls through on the score."""
    profile = _profile()
    row = {
        "sha256": "noxcd",
        "quality_score": 2.5,
        "cascade_decision": None,
        "cascade_final_stage": None,
        "cascade_attempts": None,
    }
    manifest = build_manifest_row(row, profile)
    assert manifest["decision"] == "publish"
    assert manifest["blockers"] == {}
    assert manifest["cascade_final_stage"] is None


def test_run_gate_reads_jsonl_and_writes_manifest(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.toml"
    profile_path.write_text(textwrap.dedent("""
        name = "t"
        version = "1.0.0"
        created_at = "2026-05-22"
        description = ""
        [grade_boundaries]
        excellent = 2.5
        good = 1.5
        fair = 0.5
        [decision]
        t_publish = 2.0
        t_reject = 0.5
        [blockers]
        disable = []
    """), encoding="utf-8")

    bench_path = tmp_path / "bench.jsonl"
    rows = [
        _bench_row(quality_score=2.3),                   # publish
        _bench_row(quality_score=1.0),                   # review (grey)
        _bench_row(quality_score=0.1),                   # reject (score)
    ]
    rows[0]["sha256"] = "row0"
    rows[1]["sha256"] = "row1"
    rows[2]["sha256"] = "row2"
    bench_path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    out_path = tmp_path / "manifest.jsonl"
    summary = run_gate(bench_path, out_path, profile_path)

    assert summary["num_rows"] == 3
    assert summary["by_decision"] == {"publish": 1, "review": 1, "reject": 1}

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    assert [m["decision"] for m in parsed] == ["publish", "review", "reject"]
    assert [m["doc_id"] for m in parsed] == ["row0", "row1", "row2"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/bench/test_release_gate.py -v`
Expected: 3 new failures, `ImportError: cannot import name 'build_manifest_row'` / `'run_gate'`.

- [ ] **Step 3: Implement `build_manifest_row` and `run_gate`**

Append to `packages/pdfsys-bench/src/pdfsys_bench/release_gate.py`:

```python
import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable

# v1 reserves these schema keys for future Layer-3 / page-level work.
# Always emitted as null so consumers can rely on their presence.
_NULL_RESERVED_FIELDS = (
    "page_quality_p05",
    "page_quality_min",
    "bad_page_ratio",
    "visual_alignment_score",
    "consensus_score",
)

# LLM review fields — populated later by llm_review.py if it runs.
_NULL_LLM_FIELDS = (
    "quality_score_llm",
    "quality_reason_llm",
    "quality_model_llm",
)


def build_manifest_row(row: dict, profile: ThresholdProfile) -> dict:
    """Build one manifest row from one bench-loop row."""
    decision, grade, reasons = decide(row, profile)
    blockers = _final_blockers(row)
    score = row.get("quality_score")

    manifest: dict = {
        "doc_id": row.get("sha256"),
        "decision": decision,
        "doc_quality_score": score,
        "doc_quality_grade": grade,
        "blockers": blockers,
        "reasons": reasons,
        "cascade_final_stage": row.get("cascade_final_stage"),
    }
    for k in _NULL_RESERVED_FIELDS:
        manifest[k] = None
    manifest["scorer_version"] = SCORER_VERSION
    manifest["threshold_profile"] = profile.identifier
    for k in _NULL_LLM_FIELDS:
        manifest[k] = None
    return manifest


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e


def run_gate(
    bench_jsonl: str | Path,
    out_path: str | Path,
    profile_path: str | Path,
) -> dict:
    """Read bench JSONL, apply the gate, write the release manifest.

    Returns a summary dict with ``num_rows`` and ``by_decision``.
    """
    bench_jsonl = Path(bench_jsonl)
    out_path = Path(out_path)
    profile = load_profile(profile_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    num_rows = 0
    with out_path.open("w", encoding="utf-8") as out:
        for row in _iter_jsonl(bench_jsonl):
            manifest = build_manifest_row(row, profile)
            out.write(json.dumps(manifest, ensure_ascii=False) + "\n")
            counts[manifest["decision"]] += 1
            num_rows += 1

    return {
        "num_rows": num_rows,
        "by_decision": dict(counts),
        "out_path": str(out_path),
        "profile": profile.identifier,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdfsys_bench.release_gate",
        description="Apply Layer-4 release decisions to a bench JSONL.",
    )
    p.add_argument("--bench-jsonl", required=True, type=Path,
                   help="Input bench JSONL from `python -m pdfsys_bench`")
    p.add_argument("--out", required=True, type=Path,
                   help="Output release manifest JSONL path")
    p.add_argument("--profile", required=True, type=Path,
                   help="Path to a TOML threshold profile")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    summary = run_gate(args.bench_jsonl, args.out, args.profile)
    print(f"[release-gate] profile      = {summary['profile']}")
    print(f"[release-gate] num_rows     = {summary['num_rows']}")
    print(f"[release-gate] by_decision  = {summary['by_decision']}")
    print(f"[release-gate] manifest at  = {summary['out_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/bench/test_release_gate.py -v`
Expected: 15 passed.

- [ ] **Step 5: CLI smoke test**

Run:
```bash
uv run python -m pdfsys_bench.release_gate --help
```
Expected: usage block listing `--bench-jsonl`, `--out`, `--profile`.

- [ ] **Step 6: Lint**

Run: `uv run ruff check packages/pdfsys-bench/src/pdfsys_bench/release_gate.py tests/bench/test_release_gate.py`
Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add packages/pdfsys-bench/src/pdfsys_bench/release_gate.py tests/bench/test_release_gate.py
git commit -m "feat(bench): release-gate run_gate() + CLI → release_manifest.jsonl"
```

---

## Task 4: Calibration directory scaffold

Documentation + empty data files. No tests; this is infrastructure.

**Files:**
- Create: `packages/pdfsys-bench/calibration/README.md`
- Create: `packages/pdfsys-bench/calibration/labels.jsonl` (empty)

- [ ] **Step 1: Create `labels.jsonl` (empty)**

Run:
```bash
mkdir -p packages/pdfsys-bench/calibration
: > packages/pdfsys-bench/calibration/labels.jsonl
```

- [ ] **Step 2: Create `README.md`**

Create `packages/pdfsys-bench/calibration/README.md`:

```markdown
# Release-gate calibration

Ground-truth labels and threshold profiles for the Layer-4 release gate.

Spec: `docs/superpowers/specs/2026-05-22-release-gate-layer4-design.md`.

## Layout

- `profiles/<name>.toml` — threshold profiles consumed by `release_gate.py`.
- `labels.jsonl` — append-only JSONL; one row per labeling decision.
  Latest-wins on `doc_id` when re-read.

## Label schema

```json
{
  "doc_id": "<sha256>",
  "doc_quality": 2,
  "doc_publishable": true,
  "severity": "none",
  "issue_flags": [],
  "note": "",
  "labeled_by": "<user>",
  "labeled_at": "<iso8601>",
  "source": "human",
  "draft_score_llm": 2.0,
  "draft_reason_llm": "..."
}
```

- `doc_quality`: 0–3 (FinePDFs rubric).
- `doc_publishable`: `true | false | null`. `null` for LLM-draft rows
  awaiting human review.
- `severity`: `none | minor | major | critical` (set when `doc_publishable=false`).
- `issue_flags`: subset of `["garbage_text", "repetition", "encoding_issue",
  "missing_content", "broken_table", "reading_order"]`.
- `source`: `human | llm_draft` — humans override drafts on the same `doc_id`.

## Workflow

1. Seed drafts:
   ```
   uv run python -m pdfsys_bench.llm_review \
       --manifest out/release_manifest.jsonl \
       --markdown-dir out/viz_final/markdown \
       --llm-scope all
   ```
   Then export to `labels.jsonl` with `source=llm_draft` and
   `doc_publishable=null`.

2. Human review in the viz site (recommended) or by appending rows to
   `labels.jsonl` directly. New rows for the same `doc_id` override
   prior ones.

3. Once ≥ 50 / 150 rows are human-labeled, re-fit:
   ```
   uv run python -m pdfsys_bench.fit_profile \
       --bench-jsonl out/bench_full.jsonl \
       --labels packages/pdfsys-bench/calibration/labels.jsonl \
       --out packages/pdfsys-bench/calibration/profiles/default-v1.toml
   ```

## Re-fitting guidance

Bump `version` (e.g. `0.1.0` → `0.2.0`) whenever thresholds change.
Profile identifiers (`<name>@<version>`) are written into every manifest
row's `threshold_profile` field so historical decisions stay traceable.
```

- [ ] **Step 3: Verify files exist**

Run: `ls -la packages/pdfsys-bench/calibration/`
Expected: `README.md`, `labels.jsonl` (empty), `profiles/` directory.

- [ ] **Step 4: Commit**

```bash
git add packages/pdfsys-bench/calibration/README.md packages/pdfsys-bench/calibration/labels.jsonl
git commit -m "docs(bench): release-gate calibration directory + README"
```

---

## Task 5: `llm_review.py` — scope-controlled offline scoring

Reads the manifest, scores selected rows via `LlmQualityScorer`, writes `quality_*_llm` fields back in place. Resume via append-only checkpoint.

**Files:**
- Create: `packages/pdfsys-bench/src/pdfsys_bench/llm_review.py`
- Create: `tests/bench/test_llm_review.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/bench/test_llm_review.py`:

```python
"""LLM external-review tests.

The LLM client is mocked — these tests never hit the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pdfsys_bench.llm_review import filter_manifest_rows, run_review


def _manifest_row(
    doc_id: str,
    decision: str = "publish",
    markdown_path: str = "md/x.md",
) -> dict:
    return {
        "doc_id": doc_id,
        "decision": decision,
        "doc_quality_score": 2.0,
        "doc_quality_grade": "good",
        "blockers": {},
        "reasons": [],
        "cascade_final_stage": "mupdf",
        "page_quality_p05": None,
        "page_quality_min": None,
        "bad_page_ratio": None,
        "visual_alignment_score": None,
        "consensus_score": None,
        "scorer_version": "release-gate-v0.1",
        "threshold_profile": "default-v1@0.1.0",
        "quality_score_llm": None,
        "quality_reason_llm": None,
        "quality_model_llm": None,
        # Auxiliary — kept from the bench JSONL by run_gate's caller in
        # the real pipeline. Tests inject it directly.
        "_markdown_path": markdown_path,
    }


def test_filter_scope_all_keeps_every_row() -> None:
    rows = [_manifest_row("a", "publish"), _manifest_row("b", "review"),
            _manifest_row("c", "reject")]
    kept = list(filter_manifest_rows(rows, scope="all"))
    assert [r["doc_id"] for r in kept] == ["a", "b", "c"]


def test_filter_scope_review_keeps_only_review_rows() -> None:
    rows = [_manifest_row("a", "publish"), _manifest_row("b", "review"),
            _manifest_row("c", "reject")]
    kept = list(filter_manifest_rows(rows, scope="review"))
    assert [r["doc_id"] for r in kept] == ["b"]


def test_filter_rejects_unknown_scope() -> None:
    with pytest.raises(ValueError, match="scope"):
        list(filter_manifest_rows([], scope="bogus"))


def test_run_review_patches_llm_fields(tmp_path: Path) -> None:
    md_dir = tmp_path / "md"
    md_dir.mkdir()
    (md_dir / "a.md").write_text("Some clean markdown.", encoding="utf-8")

    manifest = tmp_path / "manifest.jsonl"
    rows = [_manifest_row("a", "publish", markdown_path="a.md")]
    manifest.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    scorer = MagicMock()
    fake_result = MagicMock()
    fake_result.score = 2.5
    fake_result.reason = "clean"
    fake_result.model = "mimo-v2.5-pro"
    fake_result.parse_error = None
    scorer.score.return_value = fake_result
    scorer.client.config.model = "mimo-v2.5-pro"

    summary = run_review(
        manifest_path=manifest,
        markdown_dir=md_dir,
        scope="all",
        scorer=scorer,
        workers=1,
    )

    assert summary["num_scored"] == 1
    assert scorer.score.call_count == 1
    patched = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    assert patched["quality_score_llm"] == 2.5
    assert patched["quality_reason_llm"] == "clean"
    assert patched["quality_model_llm"] == "mimo-v2.5-pro"
    assert patched["decision"] == "publish"  # untouched


def test_run_review_resume_skips_already_scored(tmp_path: Path) -> None:
    md_dir = tmp_path / "md"
    md_dir.mkdir()
    (md_dir / "a.md").write_text("aaa", encoding="utf-8")
    (md_dir / "b.md").write_text("bbb", encoding="utf-8")

    manifest = tmp_path / "manifest.jsonl"
    rows = [
        _manifest_row("a", "publish", markdown_path="a.md"),
        _manifest_row("b", "publish", markdown_path="b.md"),
    ]
    manifest.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    # Pre-populate the checkpoint for "a".
    ckpt = tmp_path / "manifest.jsonl.llm.jsonl"
    ckpt.write_text(json.dumps({
        "doc_id": "a",
        "quality_score_llm": 2.5,
        "quality_reason_llm": "from ckpt",
        "quality_model_llm": "mimo-v2.5-pro",
    }) + "\n", encoding="utf-8")

    scorer = MagicMock()
    fake_result = MagicMock()
    fake_result.score = 1.0
    fake_result.reason = "fresh"
    fake_result.model = "mimo-v2.5-pro"
    fake_result.parse_error = None
    scorer.score.return_value = fake_result
    scorer.client.config.model = "mimo-v2.5-pro"

    summary = run_review(
        manifest_path=manifest,
        markdown_dir=md_dir,
        scope="all",
        scorer=scorer,
        workers=1,
        resume=True,
    )

    # Only "b" should have actually been scored
    assert scorer.score.call_count == 1
    assert summary["num_scored"] == 1
    assert summary["num_resumed"] == 1

    patched = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    by_id = {r["doc_id"]: r for r in patched}
    assert by_id["a"]["quality_score_llm"] == 2.5   # from checkpoint
    assert by_id["a"]["quality_reason_llm"] == "from ckpt"
    assert by_id["b"]["quality_score_llm"] == 1.0   # fresh
    assert by_id["b"]["quality_reason_llm"] == "fresh"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/bench/test_llm_review.py -v`
Expected: 5 failures with `ImportError: cannot import name 'filter_manifest_rows'`.

- [ ] **Step 3: Implement `llm_review.py`**

Create `packages/pdfsys-bench/src/pdfsys_bench/llm_review.py`:

```python
"""Offline LLM external-review for release manifests.

Reads a release manifest JSONL produced by ``release_gate.py``, scores
selected rows with the LLM scorer, and writes ``quality_*_llm`` fields
back into the same file. Decision fields are NEVER modified — the LLM
is strictly an external reviewer, not part of the decision rule.

Scope:
* ``--llm-scope all``     — score every row (benchmark phase, default v1)
* ``--llm-scope review``  — score only rows with ``decision == "review"``

Resume via ``<manifest>.llm.jsonl``: an append-only checkpoint of each
``{doc_id, quality_score_llm, ...}`` record. With ``--resume``, rows
whose ``doc_id`` already appears in the checkpoint are skipped.

Hard architectural rule (enforced by convention, see the spec):
``llm_review`` MUST NOT be imported by ``loop.py``, ``cascade.py``, or
``release_gate.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .quality_llm import LlmQualityScorer

_VALID_SCOPES = ("all", "review")
_PATCH_FIELDS = ("quality_score_llm", "quality_reason_llm", "quality_model_llm")


def filter_manifest_rows(
    rows: Iterable[dict],
    *,
    scope: str,
) -> Iterator[dict]:
    """Yield only the rows the given scope applies to."""
    if scope not in _VALID_SCOPES:
        raise ValueError(f"scope must be one of {_VALID_SCOPES}; got {scope!r}")
    for row in rows:
        if scope == "all" or row.get("decision") == "review":
            yield row


def _load_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
    return rows


def _write_manifest(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    by_id: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sha = rec.get("doc_id")
            if sha:
                by_id[sha] = rec
    return by_id


def _append_checkpoint(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def _resolve_markdown_path(row: dict, markdown_dir: Path) -> Path | None:
    """Resolve a manifest row to its markdown file.

    Prefers an explicit ``_markdown_path`` (relative to ``markdown_dir``)
    when present; otherwise falls back to ``<doc_id>.md``.
    """
    rel = row.get("_markdown_path") or f"{row['doc_id']}.md"
    candidate = markdown_dir / rel
    return candidate if candidate.exists() else None


def _score_row(
    scorer: LlmQualityScorer,
    row: dict,
    markdown_dir: Path,
) -> dict[str, Any]:
    md_path = _resolve_markdown_path(row, markdown_dir)
    if md_path is None:
        return {
            "doc_id": row["doc_id"],
            "quality_score_llm": None,
            "quality_reason_llm": f"markdown not found for {row['doc_id']}",
            "quality_model_llm": scorer.client.config.model,
            "quality_parse_error_llm": "missing markdown",
        }
    text = md_path.read_text(encoding="utf-8", errors="replace")
    result = scorer.score(text)
    return {
        "doc_id": row["doc_id"],
        "quality_score_llm": result.score,
        "quality_reason_llm": result.reason,
        "quality_model_llm": result.model,
        "quality_parse_error_llm": result.parse_error,
    }


def run_review(
    *,
    manifest_path: str | Path,
    markdown_dir: str | Path,
    scope: str,
    scorer: Any = None,
    workers: int = 4,
    resume: bool = False,
) -> dict[str, Any]:
    """Run LLM review against a manifest in place.

    ``scorer`` defaults to a fresh :class:`LlmQualityScorer` (which reads
    ``.env``). Tests pass a mock.
    """
    manifest_path = Path(manifest_path)
    markdown_dir = Path(markdown_dir)
    rows = _load_manifest(manifest_path)

    scorer = scorer if scorer is not None else LlmQualityScorer()
    ckpt_path = manifest_path.with_suffix(manifest_path.suffix + ".llm.jsonl")
    checkpoint = _load_checkpoint(ckpt_path) if resume else {}

    in_scope = list(filter_manifest_rows(rows, scope=scope))
    if resume:
        todo = [r for r in in_scope if r["doc_id"] not in checkpoint]
    else:
        todo = in_scope

    by_id: dict[str, dict[str, Any]] = dict(checkpoint)  # start with resumed

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_score_row, scorer, r, markdown_dir): r
                for r in todo
            }
            for fut in as_completed(futures):
                rec = fut.result()
                by_id[rec["doc_id"]] = rec
                _append_checkpoint(ckpt_path, rec)

    # Patch the manifest in place.
    for row in rows:
        rec = by_id.get(row["doc_id"])
        if rec is None:
            continue
        for k in _PATCH_FIELDS:
            row[k] = rec.get(k)

    _write_manifest(manifest_path, rows)

    return {
        "num_rows": len(rows),
        "num_in_scope": len(in_scope),
        "num_resumed": len(in_scope) - len(todo) if resume else 0,
        "num_scored": len(todo),
        "checkpoint": str(ckpt_path),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdfsys_bench.llm_review",
        description="Offline LLM review for release manifests (external; not in decision rule).",
    )
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--markdown-dir", required=True, type=Path)
    p.add_argument("--llm-scope", choices=_VALID_SCOPES, default="all",
                   help="'all' (default, benchmark phase) or 'review' (production phase)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", action="store_true",
                   help="Skip rows already in <manifest>.llm.jsonl")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    summary = run_review(
        manifest_path=args.manifest,
        markdown_dir=args.markdown_dir,
        scope=args.llm_scope,
        workers=args.workers,
        resume=args.resume,
    )
    print(f"[llm-review] scope        = {args.llm_scope}")
    print(f"[llm-review] num_rows     = {summary['num_rows']}")
    print(f"[llm-review] num_in_scope = {summary['num_in_scope']}")
    print(f"[llm-review] num_resumed  = {summary['num_resumed']}")
    print(f"[llm-review] num_scored   = {summary['num_scored']}")
    print(f"[llm-review] checkpoint   = {summary['checkpoint']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/bench/test_llm_review.py -v`
Expected: 5 passed.

- [ ] **Step 5: Verify the module imports cleanly without network**

Run: `uv run python -c "from pdfsys_bench import llm_review; print(llm_review.filter_manifest_rows.__doc__)"`
Expected: docstring prints; no network call (LlmQualityScorer is only constructed on `run_review` default).

- [ ] **Step 6: Lint**

Run: `uv run ruff check packages/pdfsys-bench/src/pdfsys_bench/llm_review.py tests/bench/test_llm_review.py`
Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add packages/pdfsys-bench/src/pdfsys_bench/llm_review.py tests/bench/test_llm_review.py
git commit -m "feat(bench): offline llm_review with --llm-scope + resume checkpoint"
```

---

## Task 6: `fit_profile.py` — threshold search from labels

Grid search over `(t_publish, t_reject)` against `calibration/labels.jsonl`. Optimizes the spec's objective: maximize publish-of-truly-publishable subject to `false_publish_rate ≤ 0.05` and `review_rate ≤ 0.30`.

**Files:**
- Create: `packages/pdfsys-bench/src/pdfsys_bench/fit_profile.py`
- Create: `tests/bench/test_fit_profile.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/bench/test_fit_profile.py`:

```python
"""Threshold-fitting tests on synthetic labels."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pdfsys_bench.fit_profile import (
    FitResult,
    evaluate_thresholds,
    fit_thresholds,
)


def _bench(path: Path, items: list[tuple[str, float]]) -> None:
    """items = [(doc_id, quality_score)]"""
    path.write_text(
        "\n".join(json.dumps({"sha256": d, "quality_score": s}) for d, s in items),
        encoding="utf-8",
    )


def _labels(path: Path, items: list[tuple[str, bool]]) -> None:
    """items = [(doc_id, publishable)] — all 'source=human'."""
    path.write_text(
        "\n".join(
            json.dumps({
                "doc_id": d, "doc_publishable": pub,
                "doc_quality": 3 if pub else 1,
                "source": "human", "severity": "none" if pub else "minor",
                "issue_flags": [], "note": "",
                "labeled_by": "t", "labeled_at": "2026-05-22T00:00:00",
            })
            for d, pub in items
        ),
        encoding="utf-8",
    )


def test_evaluate_thresholds_counts_correctly(tmp_path: Path) -> None:
    bench = tmp_path / "bench.jsonl"
    labels = tmp_path / "labels.jsonl"
    _bench(bench, [("a", 2.5), ("b", 1.0), ("c", 0.2)])
    _labels(labels, [("a", True), ("b", True), ("c", False)])

    res = evaluate_thresholds(
        bench_jsonl=bench, labels_jsonl=labels,
        t_publish=2.0, t_reject=0.5,
    )
    # a: score 2.5 >= 2.0 → publish (label publishable=True) → TP publish
    # b: score 1.0 in grey → review
    # c: score 0.2 < 0.5  → reject (label publishable=False) → correct reject
    assert res["counts"]["publish"] == 1
    assert res["counts"]["review"] == 1
    assert res["counts"]["reject"] == 1
    assert res["false_publish_rate"] == 0.0
    assert res["review_rate"] == pytest.approx(1 / 3)
    assert res["publish_of_publishable"] == 1


def test_fit_returns_best_thresholds_under_constraints(tmp_path: Path) -> None:
    bench = tmp_path / "bench.jsonl"
    labels = tmp_path / "labels.jsonl"
    # Clear bimodal — score >=2 always publishable, score <=1 never publishable.
    _bench(bench, [
        ("a", 2.9), ("b", 2.5), ("c", 2.1),     # all publishable
        ("d", 0.9), ("e", 0.6), ("f", 0.1),     # all unpublishable
    ])
    _labels(labels, [
        ("a", True), ("b", True), ("c", True),
        ("d", False), ("e", False), ("f", False),
    ])

    fit: FitResult = fit_thresholds(
        bench_jsonl=bench, labels_jsonl=labels,
        false_publish_max=0.05, review_rate_max=0.30,
    )
    assert fit.t_reject < fit.t_publish
    assert fit.report["false_publish_rate"] <= 0.05
    assert fit.report["review_rate"] <= 0.30
    # With clean bimodality, all publishable should pass.
    assert fit.report["publish_of_publishable"] == 3


def test_fit_raises_when_no_feasible_thresholds(tmp_path: Path) -> None:
    bench = tmp_path / "bench.jsonl"
    labels = tmp_path / "labels.jsonl"
    # Pathological: high-score docs all labeled unpublishable, no feasible cutoff.
    _bench(bench, [("a", 2.9), ("b", 2.8), ("c", 2.7)])
    _labels(labels, [("a", False), ("b", False), ("c", False)])

    with pytest.raises(RuntimeError, match="no feasible"):
        fit_thresholds(
            bench_jsonl=bench, labels_jsonl=labels,
            false_publish_max=0.0, review_rate_max=0.0,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/bench/test_fit_profile.py -v`
Expected: 3 failures with `ImportError`.

- [ ] **Step 3: Implement `fit_profile.py`**

Create `packages/pdfsys-bench/src/pdfsys_bench/fit_profile.py`:

```python
"""Threshold fitting for release-gate profiles.

Reads bench JSONL + ``calibration/labels.jsonl``, then grid-searches
``(t_publish, t_reject)`` to satisfy the constraints in the spec:

* ``false_publish_rate <= false_publish_max`` (default 0.05)
* ``review_rate         <= review_rate_max``   (default 0.30)

Among feasible points, picks the one that maximizes
``publish_of_publishable`` (true positives in the publish bucket),
breaking ties by smaller ``review_rate``.

Emits a fitted TOML profile (or just reports the chosen thresholds when
``--out`` is omitted).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_FALSE_PUBLISH_MAX = 0.05
_DEFAULT_REVIEW_RATE_MAX = 0.30
_GRID_STEP = 0.1


@dataclass(slots=True)
class FitResult:
    t_publish: float
    t_reject: float
    report: dict[str, Any]


def _load_scores(bench_jsonl: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with bench_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sha = row.get("sha256") or row.get("doc_id")
            score = row.get("quality_score")
            if sha and score is not None:
                out[sha] = float(score)
    return out


def _load_human_labels(labels_jsonl: Path) -> dict[str, bool]:
    """Latest-wins across the file. Only ``source=human`` rows count;
    LLM-draft rows are ignored. ``doc_publishable=null`` rows are also
    ignored."""
    by_id: dict[str, dict[str, Any]] = {}
    with labels_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sha = rec.get("doc_id")
            if not sha:
                continue
            by_id[sha] = rec  # later rows overwrite
    return {
        sha: bool(rec["doc_publishable"])
        for sha, rec in by_id.items()
        if rec.get("source") == "human" and rec.get("doc_publishable") is not None
    }


def evaluate_thresholds(
    *,
    bench_jsonl: Path,
    labels_jsonl: Path,
    t_publish: float,
    t_reject: float,
) -> dict[str, Any]:
    """Score one ``(t_publish, t_reject)`` point against labelled bench rows.

    Only rows present in both files contribute. Ignores Layer-1 blockers
    — we're fitting score thresholds, not the blocker logic.
    """
    scores = _load_scores(bench_jsonl)
    labels = _load_human_labels(labels_jsonl)
    common = sorted(scores.keys() & labels.keys())
    n = len(common)
    if n == 0:
        return {
            "counts": {"publish": 0, "review": 0, "reject": 0},
            "false_publish_rate": 0.0,
            "review_rate": 0.0,
            "publish_of_publishable": 0,
            "n": 0,
        }

    counts = {"publish": 0, "review": 0, "reject": 0}
    false_publish = 0
    publish_of_publishable = 0
    n_publishable = sum(1 for sha in common if labels[sha])

    for sha in common:
        s = scores[sha]
        pub_label = labels[sha]
        if s >= t_publish:
            decision = "publish"
        elif s < t_reject:
            decision = "reject"
        else:
            decision = "review"
        counts[decision] += 1
        if decision == "publish":
            if pub_label:
                publish_of_publishable += 1
            else:
                false_publish += 1

    return {
        "counts": counts,
        "false_publish_rate": false_publish / n_publishable if n_publishable else 0.0,
        "review_rate": counts["review"] / n,
        "publish_of_publishable": publish_of_publishable,
        "n": n,
    }


def fit_thresholds(
    *,
    bench_jsonl: Path,
    labels_jsonl: Path,
    false_publish_max: float = _DEFAULT_FALSE_PUBLISH_MAX,
    review_rate_max: float = _DEFAULT_REVIEW_RATE_MAX,
    step: float = _GRID_STEP,
) -> FitResult:
    """Grid-search the (t_publish, t_reject) plane.

    Search range: t_publish in [step, 3.0], t_reject in [0, t_publish - step].
    """
    best: FitResult | None = None
    t = 0.0
    candidates: list[FitResult] = []
    while t <= 3.0 + 1e-9:
        t_publish = round(t, 4)
        u = 0.0
        while u < t_publish - 1e-9:
            t_reject = round(u, 4)
            if t_publish > 0 and t_reject < t_publish:
                report = evaluate_thresholds(
                    bench_jsonl=bench_jsonl,
                    labels_jsonl=labels_jsonl,
                    t_publish=t_publish,
                    t_reject=t_reject,
                )
                if (
                    report["false_publish_rate"] <= false_publish_max
                    and report["review_rate"] <= review_rate_max
                ):
                    candidates.append(FitResult(
                        t_publish=t_publish, t_reject=t_reject, report=report,
                    ))
            u += step
        t += step

    if not candidates:
        raise RuntimeError(
            "no feasible thresholds satisfy the constraints "
            f"(false_publish_max={false_publish_max}, review_rate_max={review_rate_max})"
        )

    # Highest publish_of_publishable, ties broken by lower review_rate.
    candidates.sort(
        key=lambda c: (
            -c.report["publish_of_publishable"],
            c.report["review_rate"],
        )
    )
    best = candidates[0]
    return best


def _emit_profile(
    out_path: Path,
    *,
    name: str,
    version: str,
    description: str,
    t_publish: float,
    t_reject: float,
) -> None:
    body = (
        f'name = "{name}"\n'
        f'version = "{version}"\n'
        f'created_at = "{_today()}"\n'
        f'description = "{description}"\n'
        "\n"
        "[grade_boundaries]\n"
        "excellent = 2.5\n"
        "good = 1.5\n"
        "fair = 0.5\n"
        "\n"
        "[decision]\n"
        f"t_publish = {t_publish}\n"
        f"t_reject = {t_reject}\n"
        "\n"
        "[blockers]\n"
        "disable = []\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")


def _today() -> str:
    import datetime
    return datetime.date.today().isoformat()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdfsys_bench.fit_profile",
        description="Fit release-gate thresholds from calibration labels.",
    )
    p.add_argument("--bench-jsonl", required=True, type=Path)
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None,
                   help="If given, write a fitted TOML profile here")
    p.add_argument("--name", default="default-v1")
    p.add_argument("--version", default="0.2.0",
                   help="Bump this whenever you re-fit")
    p.add_argument("--description", default="fitted from calibration labels")
    p.add_argument("--false-publish-max", type=float,
                   default=_DEFAULT_FALSE_PUBLISH_MAX)
    p.add_argument("--review-rate-max", type=float,
                   default=_DEFAULT_REVIEW_RATE_MAX)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    fit = fit_thresholds(
        bench_jsonl=args.bench_jsonl,
        labels_jsonl=args.labels,
        false_publish_max=args.false_publish_max,
        review_rate_max=args.review_rate_max,
    )
    print(f"[fit-profile] t_publish              = {fit.t_publish}")
    print(f"[fit-profile] t_reject               = {fit.t_reject}")
    print(f"[fit-profile] false_publish_rate     = {fit.report['false_publish_rate']:.3f}")
    print(f"[fit-profile] review_rate            = {fit.report['review_rate']:.3f}")
    print(f"[fit-profile] publish_of_publishable = {fit.report['publish_of_publishable']}")
    print(f"[fit-profile] counts                 = {fit.report['counts']}")
    if args.out:
        _emit_profile(
            args.out,
            name=args.name,
            version=args.version,
            description=args.description,
            t_publish=fit.t_publish,
            t_reject=fit.t_reject,
        )
        print(f"[fit-profile] wrote                  = {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/bench/test_fit_profile.py -v`
Expected: 3 passed.

- [ ] **Step 5: Lint**

Run: `uv run ruff check packages/pdfsys-bench/src/pdfsys_bench/fit_profile.py tests/bench/test_fit_profile.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add packages/pdfsys-bench/src/pdfsys_bench/fit_profile.py tests/bench/test_fit_profile.py
git commit -m "feat(bench): release-gate fit_profile — grid search t_publish/t_reject"
```

---

## Task 7: viz server — `POST /api/label` + `GET /api/labels`

Mirror the existing `_load_badcases` / `_append_badcase` pattern with a separate `LABELS_PATH`. No new test file — viz_server has module-level globals that resist isolated testing, and the existing module ships untested. Verify by curl smoke test.

**Files:**
- Modify: `packages/pdfsys-bench/viz/viz_server.py`

- [ ] **Step 1: Read the current state**

Run: `wc -l packages/pdfsys-bench/viz/viz_server.py`
Note the line count — confirms the file is still at the version analyzed in this plan.

- [ ] **Step 2: Add the label helpers and constants**

At the top of `packages/pdfsys-bench/viz/viz_server.py`, just below the existing `BADCASES_PATH` line, add:

```python
LABELS_PATH = BUNDLE_DIR / "labels.jsonl"
VALID_SEVERITIES = ("none", "minor", "major", "critical")
VALID_ISSUE_FLAGS = (
    "garbage_text", "repetition", "encoding_issue",
    "missing_content", "broken_table", "reading_order",
)
```

Below `_append_badcase`, add:

```python
def _load_labels() -> list[dict]:
    """Load all label records, latest-per-doc_id wins."""
    if not LABELS_PATH.exists():
        return []
    latest: dict[str, dict] = {}
    with LABELS_PATH.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"[viz_server] WARN: labels line {line_no} not JSON", file=sys.stderr)
                continue
            sha = rec.get("doc_id")
            if not sha:
                continue
            prev = latest.get(sha)
            if prev is None or rec.get("labeled_at", "") >= prev.get("labeled_at", ""):
                latest[sha] = rec
    return list(latest.values())


def _append_label(rec: dict) -> None:
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    with LABELS_PATH.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
```

- [ ] **Step 3: Wire the GET endpoint**

In `VizHandler.do_GET`, just before the `self._serve_static()` fallback, add:

```python
        if self.path == "/api/labels":
            self._send_json(HTTPStatus.OK, {"labels": _load_labels()})
            return
```

The full method should now read:

```python
    def do_GET(self) -> None:
        if self.path == "/api/badcases":
            self._send_json(HTTPStatus.OK, {"badcases": _load_badcases()})
            return
        if self.path == "/api/labels":
            self._send_json(HTTPStatus.OK, {"labels": _load_labels()})
            return
        self._serve_static()
```

- [ ] **Step 4: Wire the POST endpoint**

In `VizHandler.do_POST`, change the early-return guard from `if self.path != "/api/badcase":` to also accept `/api/label`. Replace the existing method body with:

```python
    def do_POST(self) -> None:
        if self.path == "/api/badcase":
            return self._handle_post_badcase()
        if self.path == "/api/label":
            return self._handle_post_label()
        self._send_text(HTTPStatus.NOT_FOUND, "no such endpoint")
```

Move the existing badcase POST body into `_handle_post_badcase`, then add `_handle_post_label`:

```python
    def _handle_post_badcase(self) -> None:
        body = self._read_json_body()
        if not isinstance(body, dict):
            self._send_text(HTTPStatus.BAD_REQUEST, "body must be JSON object")
            return
        sha = body.get("sha256")
        if not isinstance(sha, str) or not sha:
            self._send_text(HTTPStatus.BAD_REQUEST, "missing sha256")
            return
        stage = body.get("stage", "overall")
        if stage not in VALID_STAGES:
            self._send_text(HTTPStatus.BAD_REQUEST, f"stage must be one of {VALID_STAGES}")
            return
        tags = body.get("tags") or []
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            self._send_text(HTTPStatus.BAD_REQUEST, "tags must be list[str]")
            return
        note = body.get("note", "")
        if not isinstance(note, str):
            self._send_text(HTTPStatus.BAD_REQUEST, "note must be str")
            return
        rec = {
            "sha256": sha,
            "is_bad": True,
            "stage": stage,
            "tags": tags,
            "note": note[:1000],
            "flagged_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds"),
            "flagged_by": USER,
        }
        _append_badcase(rec)
        self._send_json(HTTPStatus.OK, rec)

    def _handle_post_label(self) -> None:
        body = self._read_json_body()
        if not isinstance(body, dict):
            self._send_text(HTTPStatus.BAD_REQUEST, "body must be JSON object")
            return
        sha = body.get("doc_id")
        if not isinstance(sha, str) or not sha:
            self._send_text(HTTPStatus.BAD_REQUEST, "missing doc_id")
            return
        quality = body.get("doc_quality")
        if not (isinstance(quality, int) and 0 <= quality <= 3):
            self._send_text(HTTPStatus.BAD_REQUEST, "doc_quality must be int 0..3")
            return
        publishable = body.get("doc_publishable")
        if publishable is not None and not isinstance(publishable, bool):
            self._send_text(HTTPStatus.BAD_REQUEST, "doc_publishable must be bool or null")
            return
        severity = body.get("severity", "none")
        if severity not in VALID_SEVERITIES:
            self._send_text(HTTPStatus.BAD_REQUEST, f"severity must be one of {VALID_SEVERITIES}")
            return
        issue_flags = body.get("issue_flags") or []
        if not isinstance(issue_flags, list) or not all(f in VALID_ISSUE_FLAGS for f in issue_flags):
            self._send_text(HTTPStatus.BAD_REQUEST,
                            f"issue_flags must be a subset of {VALID_ISSUE_FLAGS}")
            return
        note = body.get("note", "")
        if not isinstance(note, str):
            self._send_text(HTTPStatus.BAD_REQUEST, "note must be str")
            return
        rec = {
            "doc_id": sha,
            "doc_quality": quality,
            "doc_publishable": publishable,
            "severity": severity,
            "issue_flags": issue_flags,
            "note": note[:1000],
            "labeled_by": USER,
            "labeled_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds"),
            "source": "human",
        }
        _append_label(rec)
        self._send_json(HTTPStatus.OK, rec)
```

- [ ] **Step 5: Update the `main()` startup banner**

In `main()` of `viz_server.py`, after the badcases print line, add:

```python
    n_labels = len(_load_labels())
    print(f"  labels.jsonl:   {LABELS_PATH} ({n_labels} active labels)")
```

- [ ] **Step 6: Smoke test by curl**

Start the server (in a separate terminal) — point it at a viz bundle that has been produced previously:

```bash
cd out/viz_final && uv run python ../../packages/pdfsys-bench/viz/viz_server.py --port 8765 --user test
```

In another terminal:

```bash
# Empty initially
curl -s http://localhost:8765/api/labels | python -m json.tool
# Expected: {"labels": []}

# POST a label
curl -s -X POST http://localhost:8765/api/label \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"abc123","doc_quality":2,"doc_publishable":true,"severity":"none","issue_flags":["broken_table"],"note":"smoke"}' \
  | python -m json.tool
# Expected: full record echoed back with labeled_by="test", source="human"

# GET reflects it
curl -s http://localhost:8765/api/labels | python -m json.tool
# Expected: {"labels": [<the record above>]}

# Validation error
curl -s -X POST http://localhost:8765/api/label \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"x","doc_quality":99}'
# Expected: "doc_quality must be int 0..3"
```

Stop the server (Ctrl-C). Remove the smoke-test record from the labels file before committing:

```bash
rm -f out/viz_final/labels.jsonl
```

- [ ] **Step 7: Lint**

Run: `uv run ruff check packages/pdfsys-bench/viz/viz_server.py`
Expected: `All checks passed!`

- [ ] **Step 8: Commit**

```bash
git add packages/pdfsys-bench/viz/viz_server.py
git commit -m "feat(viz): POST /api/label + GET /api/labels for calibration"
```

---

## Task 8: Viz UI — labeling control on the detail card

Add a minimal form to the detail card in `viz/index.html`: 0–3 radio, publishable tri-state, severity dropdown, issue-flag checkboxes, note textarea, save button. On save, POST to `/api/label` and reflect the new label in the row's UI.

**Files:**
- Modify: `packages/pdfsys-bench/viz/index.html`

- [ ] **Step 1: Locate the detail card**

Run: `grep -n "Quality\|stage.*quality\|detail.*card\|quality_score" packages/pdfsys-bench/viz/index.html | head -20`

Find the section that renders the Quality stage card inside the detail view. The labeling control goes immediately after the Quality stage card (and after the LLM Quality block that was added in this session) so reviewers see the existing scores while filling out the label.

- [ ] **Step 2: Add a `renderLabel(row)` function and form HTML**

Inside the detail-card builder where Quality / LLM-Quality blocks are rendered, append:

```html
<div class="card" id="label-card">
  <h3>Calibration label</h3>
  <div class="label-form" data-sha="${row.sha256}">
    <div class="row">
      <label>Quality (0–3):</label>
      ${[0,1,2,3].map(q =>
        `<label class="radio">
          <input type="radio" name="lbl-q-${row.sha256}" value="${q}">
          ${q}
        </label>`).join("")}
    </div>
    <div class="row">
      <label>Publishable:</label>
      <select class="lbl-pub">
        <option value="">(unknown)</option>
        <option value="true">true</option>
        <option value="false">false</option>
      </select>
    </div>
    <div class="row">
      <label>Severity:</label>
      <select class="lbl-sev">
        ${["none","minor","major","critical"].map(s =>
          `<option value="${s}">${s}</option>`).join("")}
      </select>
    </div>
    <div class="row">
      <label>Issues:</label>
      <div class="flags">
        ${["garbage_text","repetition","encoding_issue",
            "missing_content","broken_table","reading_order"].map(f =>
          `<label class="check">
            <input type="checkbox" class="lbl-flag" value="${f}"> ${f}
          </label>`).join("")}
      </div>
    </div>
    <div class="row">
      <label>Note:</label>
      <textarea class="lbl-note" rows="2" maxlength="1000"></textarea>
    </div>
    <div class="row">
      <button class="lbl-save">Save label</button>
      <span class="lbl-status"></span>
    </div>
  </div>
</div>
```

- [ ] **Step 3: Wire the save handler**

Add a click handler that gathers the form fields, posts to `/api/label`, and updates the status label. Put this near the existing `/api/badcase` POST handler:

```javascript
function wireLabelSave(card, row) {
  const form = card.querySelector(".label-form");
  if (!form) return;
  const btn = form.querySelector(".lbl-save");
  const status = form.querySelector(".lbl-status");
  btn.addEventListener("click", async () => {
    const q = form.querySelector(`input[name="lbl-q-${row.sha256}"]:checked`);
    if (!q) {
      status.textContent = "pick a quality 0..3";
      return;
    }
    const pubRaw = form.querySelector(".lbl-pub").value;
    const publishable = pubRaw === "" ? null : pubRaw === "true";
    const severity = form.querySelector(".lbl-sev").value;
    const flags = Array.from(form.querySelectorAll(".lbl-flag:checked"))
                       .map(el => el.value);
    const note = form.querySelector(".lbl-note").value;
    const body = {
      doc_id: row.sha256,
      doc_quality: Number(q.value),
      doc_publishable: publishable,
      severity, issue_flags: flags, note,
    };
    status.textContent = "saving...";
    try {
      const res = await fetch("/api/label", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const msg = await res.text();
        status.textContent = `error: ${msg}`;
        return;
      }
      const saved = await res.json();
      status.textContent = `saved at ${saved.labeled_at}`;
    } catch (err) {
      status.textContent = `network error: ${err.message}`;
    }
  });
}
```

Call `wireLabelSave(detailCard, row)` from wherever the detail card is built (look for where the existing bad-case form is wired — same pattern).

- [ ] **Step 4: Add minimal CSS for the form**

Append to the existing `<style>` block:

```css
#label-card .row { margin-bottom: 0.5em; display: flex; gap: 0.5em; align-items: center; flex-wrap: wrap; }
#label-card .flags { display: flex; gap: 0.5em; flex-wrap: wrap; }
#label-card label.radio, #label-card label.check { font-weight: normal; }
#label-card .lbl-status { color: #888; font-size: 0.9em; margin-left: 0.5em; }
```

- [ ] **Step 5: Manual smoke**

Restart the viz server pointed at a viz bundle and open `http://localhost:8765/` in a browser:

```bash
cd out/viz_final && uv run python ../../packages/pdfsys-bench/viz/viz_server.py --port 8765 --user yinz
```

Click any row to open the detail card. Fill in the form, hit "Save label". Verify the status line shows a timestamp.

```bash
cat out/viz_final/labels.jsonl
# Expected: one JSON line with the label you just submitted
```

Remove the smoke artifact before committing:

```bash
rm -f out/viz_final/labels.jsonl
```

- [ ] **Step 6: Commit**

```bash
git add packages/pdfsys-bench/viz/index.html
git commit -m "feat(viz): calibration label form on detail card (POST /api/label)"
```

---

## Task 9: End-to-end smoke + post-build notes

Run the full new pipeline against an existing bench artifact and verify the manifest distribution looks sane.

**Files:**
- Modify: `docs/superpowers/specs/2026-05-22-release-gate-layer4-design.md` (add a §15 post-build note)

- [ ] **Step 1: Confirm an existing bench JSONL is available**

Run: `ls out/`
Look for a `bench_*.jsonl` file (e.g. `bench_full.jsonl`). If none exists, run the bench loop to produce one:

```bash
uv run python -m pdfsys_bench \
    --pdf-dir packages/pdfsys-bench/omnidocbench_100/pdfs \
    --out out/bench_for_gate_smoke.jsonl \
    --limit 20 \
    --cascade
```

Set `BENCH_JSONL=out/bench_for_gate_smoke.jsonl` (or whichever file you're using) for the next steps.

- [ ] **Step 2: Run `release_gate.py`**

```bash
uv run python -m pdfsys_bench.release_gate \
    --bench-jsonl out/bench_for_gate_smoke.jsonl \
    --out out/release_manifest_smoke.jsonl \
    --profile packages/pdfsys-bench/calibration/profiles/default-v1.toml
```

Expected output: `num_rows`, `by_decision` printed; manifest file exists.

```bash
wc -l out/release_manifest_smoke.jsonl
head -1 out/release_manifest_smoke.jsonl | python -m json.tool
```

Confirm the first row has all schema fields from §5 of the spec.

- [ ] **Step 3: Check decision distribution is non-trivial**

```bash
python - <<'PY'
import collections, json
counts = collections.Counter()
for line in open("out/release_manifest_smoke.jsonl"):
    counts[json.loads(line)["decision"]] += 1
print(dict(counts))
PY
```

Expected: more than one bucket has a non-zero count (i.e. not 100% in one decision). If everything lands in one bucket, that's a flag for the placeholder thresholds — note it for the post-build §15 and proceed (it's the expected outcome before calibration).

- [ ] **Step 4: Run `llm_review.py` with `--llm-scope all` (small slice)**

Use `--limit` indirectly via picking a small bench file, or just run on the smoke manifest:

```bash
uv run python -m pdfsys_bench.llm_review \
    --manifest out/release_manifest_smoke.jsonl \
    --markdown-dir <viz-bundle-with-markdown>/markdown \
    --llm-scope all \
    --workers 4
```

Where `<viz-bundle-with-markdown>` is e.g. `out/viz_final` — the bundle that has the per-PDF markdown files for the same SHA256s. If the SHA256s in the bench JSONL don't match the bundle, point at the actual bench output's markdown dump.

Expected output: `num_scored`, `checkpoint` path printed. Verify a sample row:

```bash
head -1 out/release_manifest_smoke.jsonl | python -c "import json,sys; r=json.loads(sys.stdin.read()); print(r['quality_score_llm'], r['quality_reason_llm'][:80])"
```

Expected: non-null `quality_score_llm` and a `quality_reason_llm` snippet.

- [ ] **Step 5: Add a §15 post-build note to the spec**

Edit `docs/superpowers/specs/2026-05-22-release-gate-layer4-design.md` and append at the end:

```markdown
## 15. Post-build note (2026-05-22)

Implementation landed across 9 tasks (see `docs/superpowers/plans/2026-05-22-release-gate-layer4.md`).

**End-to-end smoke** on `out/release_manifest_smoke.jsonl`:
- `num_rows = <N>` (from `release_gate` summary)
- `by_decision = <dict>` — fill in the actual distribution
- LLM review patched `quality_*_llm` on all rows; no parse errors observed.

**Known follow-ups:**
- `default-v1@0.1.0` thresholds are hand-set; re-fit via `fit_profile.py` once
  ≥ 50 / 150 `calibration/labels.jsonl` rows are human-labeled. Bump to `0.2.0`.
- Layer 3 (visual verifier / consensus) is the next spec on the critical path
  for catching fluent-hallucination escapes that BERT alone misses.
```

Fill in the actual `<N>` and `<dict>` from Step 3 output before committing.

- [ ] **Step 6: Final lint + test sweep**

```bash
uv run ruff check packages/pdfsys-bench
uv run pytest tests/bench/ -v
```

Expected: ruff clean; pytest all green (the pre-existing cascade + quality_rules tests plus the new release_gate, llm_review, fit_profile tests).

- [ ] **Step 7: Commit the post-build note**

```bash
git add docs/superpowers/specs/2026-05-22-release-gate-layer4-design.md
git commit -m "docs(spec): release-gate Layer 4 post-build note (§15)"
```

- [ ] **Step 8: Clean up smoke artifacts**

The manifest and checkpoint files live under `out/` which is `.gitignore`d, but tidy them:

```bash
rm -f out/release_manifest_smoke.jsonl out/release_manifest_smoke.jsonl.llm.jsonl
```

---

## Self-review notes

**Spec coverage check:**
- §4 architecture (bench loop → release_gate → manifest → optional llm_review) → Tasks 1–5.
- §5 manifest schema → Task 3 (`build_manifest_row`); all required + reserved + LLM fields are emitted.
- §6 decision logic → Task 2 (`decide`); covers all 5 branches.
- §7 threshold profile + grade mapping → Tasks 1 (loader + default file), 6 (refit path).
- §8 calibration set scaffold → Task 4 (README + empty labels.jsonl).
- §9 LLM external review, scope filter, resume → Task 5.
- §10 CLI surface → covered by `__main__` in Tasks 3 (release_gate), 5 (llm_review), 6 (fit_profile).
- §11 new code listing → matches file structure section above.
- §13 acceptance criteria → exercised by tests + Task 9 smoke. The "non-trivial distribution" criterion is explicit in Task 9 Step 3.
- §14 out of scope → no tasks touch Layer 3, page-level scoring, or LLM-in-decision-rule.

**Type / signature consistency check:**
- `ThresholdProfile.identifier` used in Task 3 (`threshold_profile` field in manifest) — defined in Task 1.
- `_NULL_RESERVED_FIELDS` and `_NULL_LLM_FIELDS` are private module constants; only used inside `release_gate.py`.
- `filter_manifest_rows(rows, *, scope)` signature is consistent across Task 5 tests and implementation.
- `evaluate_thresholds(*, bench_jsonl, labels_jsonl, t_publish, t_reject) → dict` and `fit_thresholds(...) → FitResult` are consistent across tests and impl.
- Viz POST endpoints (`/api/badcase` vs `/api/label`) and JSON keys (`sha256` for bad-cases vs `doc_id` for labels) are intentionally different to match the existing semantics + the new label schema in §8 of the spec.

**Placeholder scan:** none — every step has the actual code, command, or expected output.
