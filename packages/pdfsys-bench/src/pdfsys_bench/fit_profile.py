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

    # If there are zero truly-publishable docs but we published something,
    # the false-publish rate is 1.0 (every publish was a false positive).
    # Only return 0.0 when there genuinely were no false publishes either.
    fpr = false_publish / n_publishable if n_publishable else 1.0 if false_publish else 0.0

    return {
        "counts": counts,
        "false_publish_rate": fpr,
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
    candidates: list[FitResult] = []
    t = 0.0
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
    return candidates[0]


def _today() -> str:
    import datetime

    return datetime.date.today().isoformat()


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
    print(f"[fit-profile] counts                 = {json.dumps(fit.report['counts'])}")
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
