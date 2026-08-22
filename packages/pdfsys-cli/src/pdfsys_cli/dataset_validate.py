"""Validate a ``pdfsys.page/v2`` shard against the format contract.

Everything the format promises but a writer could still get wrong: row keys,
referential integrity between the three tables, content addressing, and the
derived counters. Run it before publishing anything.

Findings come in two severities. ``error`` means the shard violates the
contract and a consumer can hit it — a marker pointing at an image that is not
there, a bbox outside the unit square, two rows claiming the same page.
``warn`` means it is legal but probably not what you meant — an image nobody
references, both image tables populated at once.

Statistics are reported separately and are never findings: an empty page or a
U+FFFD in the text is a property of the corpus, not a defect in the encoding.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from pdfsys_core import DATASET_SCHEMA_VERSION, IMAGE_REF_RE, parse_image_ref

from .dataset_writer import PAGE_SCHEMA

__all__ = ["Finding", "Report", "validate_shard"]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
#: Cap per check so one systematic mistake cannot print a million lines.
_MAX_PER_CHECK = 8


@dataclass
class Finding:
    severity: str  # "error" | "warn"
    check: str
    message: str

    def __str__(self) -> str:
        mark = "✗" if self.severity == "error" else "!"
        return f"  {mark} [{self.check}] {self.message}"


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    _counts: Counter = field(default_factory=Counter)

    def add(self, severity: str, check: str, message: str) -> None:
        self._counts[check] += 1
        if self._counts[check] <= _MAX_PER_CHECK:
            self.findings.append(Finding(severity, check, message))
        elif self._counts[check] == _MAX_PER_CHECK + 1:
            self.findings.append(
                Finding(severity, check, f"… 同类问题还有更多，仅列出前 {_MAX_PER_CHECK} 条")
            )

    def error(self, check: str, message: str) -> None:
        self.add("error", check, message)

    def warn(self, check: str, message: str) -> None:
        self.add("warn", check, message)

    @property
    def n_errors(self) -> int:
        return sum(1 for f in self.findings if f.severity == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for f in self.findings if f.severity == "warn")

    @property
    def ok(self) -> bool:
        return self.n_errors == 0


def validate_shard(shard_dir: Path, *, verify_hashes: bool = True) -> Report:
    """Check one shard. ``verify_hashes`` re-hashes every blob (slow, thorough)."""
    shard_dir = Path(shard_dir)
    report = Report()

    pages_dir = shard_dir / "pages"
    if not pages_dir.is_dir():
        report.error("layout", f"缺少 pages/ 目录: {shard_dir}")
        return report

    pages = _read(pages_dir)
    images = _read(shard_dir / "images")
    rasters = _read(shard_dir / "page_images")
    pairs = _read(shard_dir / "pairs")

    _check_schema(pages_dir, report)
    _check_keys(pages, report)
    _check_counters(pages, report)
    _check_bboxes(pages, report)
    _check_blocks(pages, report)
    _check_references(pages, images, rasters, report)
    _check_markers(pages, images, rasters, report)
    _check_media_tables(images, rasters, pages, report, verify_hashes)
    _check_pairs(pairs, pages, images, report)
    _collect_stats(pages, images, rasters, pairs, report)
    return report


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def _type_matches(want: pa.DataType, got: pa.DataType) -> bool:
    """Compare Arrow types modulo what a Parquet round trip legitimately changes.

    Two normalisations are not defects: dictionary index width is re-chosen on
    read (int8 out, int32 back), and list child fields come back named ``item``
    instead of ``element``. Everything else has to match exactly, or a strict
    reader hits a type it was not promised.
    """
    if pa.types.is_dictionary(want):
        return pa.types.is_dictionary(got) and _type_matches(
            want.value_type, got.value_type
        )
    if pa.types.is_list(want) or pa.types.is_large_list(want):
        return (
            pa.types.is_list(got) or pa.types.is_large_list(got)
        ) and _type_matches(want.value_type, got.value_type)
    if pa.types.is_struct(want):
        if not pa.types.is_struct(got) or want.num_fields != got.num_fields:
            return False
        return all(
            want.field(i).name == got.field(i).name
            and _type_matches(want.field(i).type, got.field(i).type)
            for i in range(want.num_fields)
        )
    return want == got


def _check_schema(pages_dir: Path, report: Report) -> None:
    for path in sorted(pages_dir.glob("*.parquet")):
        schema = pq.ParquetFile(str(path)).schema_arrow
        if schema.names != PAGE_SCHEMA.names:
            missing = set(PAGE_SCHEMA.names) - set(schema.names)
            extra = set(schema.names) - set(PAGE_SCHEMA.names)
            report.error(
                "schema",
                f"{path.name} 列不匹配；缺 {sorted(missing) or '无'}，多 {sorted(extra) or '无'}",
            )
            continue
        for name in PAGE_SCHEMA.names:
            want, got = PAGE_SCHEMA.field(name).type, schema.field(name).type
            if not _type_matches(want, got):
                report.error("schema", f"{path.name}.{name} 类型 {got}，应为 {want}")
        version = (schema.metadata or {}).get(b"pdfsys.schema", b"").decode()
        if version != DATASET_SCHEMA_VERSION:
            report.error(
                "schema",
                f"{path.name} 的 pdfsys.schema 是 {version or '(缺失)'}，"
                f"应为 {DATASET_SCHEMA_VERSION}",
            )


def _check_keys(pages: list[dict], report: Report) -> None:
    keys = [(p["doc_id"], p["page_index"]) for p in pages]
    for key, n in Counter(keys).items():
        if n > 1:
            report.error("key", f"{key[0][:12]}… 第 {key[1]} 页出现 {n} 次，主键必须唯一")
    if keys != sorted(keys):
        report.error("order", "行未按 (doc_id, page_index) 排序，重组文档就不再是顺序扫描")

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for p in pages:
        by_doc[p["doc_id"]].append(p)
    for doc_id, rows in by_doc.items():
        idx = sorted(r["page_index"] for r in rows)
        if idx != list(range(len(idx))):
            report.error("key", f"{doc_id[:12]}… 页码不连续: {idx[:12]}")
        declared = {r["doc_n_pages"] for r in rows}
        if len(declared) > 1:
            report.error("key", f"{doc_id[:12]}… doc_n_pages 在同一文档内不一致: {declared}")
        elif declared and (d := declared.pop()) != len(rows):
            report.error(
                "key", f"{doc_id[:12]}… doc_n_pages={d} 但只有 {len(rows)} 行"
            )
        for name in ("source_uri", "doc_lang", "doc_quality_score", "provenance"):
            if len({r[name] for r in rows}) > 1:
                report.error(
                    "denorm", f"{doc_id[:12]}… 文档级列 {name} 在各页之间不一致"
                )


def _check_counters(pages: list[dict], report: Report) -> None:
    for p in pages:
        where = f"{p['doc_id'][:12]}…/p{p['page_index']}"
        if p["text"] is not None and p["n_chars"] != len(p["text"]):
            report.error(
                "counter", f"{where} n_chars={p['n_chars']} 但 text 长 {len(p['text'])}"
            )
        blocks = p["blocks"]
        if blocks is None:
            continue
        if p["n_blocks"] != len(blocks):
            report.error("counter", f"{where} n_blocks={p['n_blocks']} 实为 {len(blocks)}")
        for name, types in (
            ("n_images", ("image", "chart")),
            ("n_tables", ("table",)),
            ("n_formulas", ("formula",)),
        ):
            actual = sum(1 for b in blocks if b["type"] in types)
            if p[name] != actual:
                report.error("counter", f"{where} {name}={p[name]} 实为 {actual}")


def _check_bboxes(pages: list[dict], report: Report) -> None:
    for p in pages:
        for b in p["blocks"] or ():
            bb = b["bbox"]
            if bb is None:
                continue
            where = f"{p['doc_id'][:12]}…/p{p['page_index']}/b{b['idx']}"
            vals = [bb[k] for k in ("x0", "y0", "x1", "y1")]
            if any(v is None for v in vals):
                report.error("bbox", f"{where} bbox 分量为 null")
            elif any(v < 0.0 or v > 1.0 for v in vals):
                report.error("bbox", f"{where} bbox 越界 [0,1]: {vals}")
            elif vals[2] < vals[0] or vals[3] < vals[1]:
                report.error("bbox", f"{where} bbox 反向: {vals}")


def _check_blocks(pages: list[dict], report: Report) -> None:
    by_doc: dict[str, set[int]] = defaultdict(set)
    seen_pages: dict[str, list[dict]] = defaultdict(list)
    for p in pages:
        seen_pages[p["doc_id"]].append(p)
        for b in p["blocks"] or ():
            if b["idx"] in by_doc[p["doc_id"]]:
                report.error(
                    "block",
                    f"{p['doc_id'][:12]}… 块下标 {b['idx']} 在文档内重复；"
                    f"mentions 靠它定位，必须文档级唯一",
                )
            by_doc[p["doc_id"]].add(b["idx"])
            if b["level"] is not None and not 1 <= b["level"] <= 6:
                report.error("block", f"{p['doc_id'][:12]}… 块 {b['idx']} level={b['level']}")

    for doc_id, rows in seen_pages.items():
        known = by_doc[doc_id]
        for p in rows:
            for b in p["blocks"] or ():
                for m in b["mentions"] or ():
                    if m not in known:
                        report.error(
                            "mention",
                            f"{doc_id[:12]}… 块 {b['idx']} 的 mentions 指向不存在的块 {m}",
                        )


def _check_references(
    pages: list[dict], images: list[dict], rasters: list[dict], report: Report
) -> None:
    have_img = {r["image_id"] for r in images}
    have_raster = {r["image_id"] for r in rasters}
    for p in pages:
        where = f"{p['doc_id'][:12]}…/p{p['page_index']}"
        listed = list(p["image_ids"] or ())
        if len(listed) != len(set(listed)):
            report.error("image_ids", f"{where} image_ids 有重复")
        for iid in listed:
            if iid not in have_img:
                report.error("ref", f"{where} image_ids 里的 {iid[:12]}… 不在 images 表")
        if p["page_image_id"] and p["page_image_id"] not in have_raster:
            report.error(
                "ref", f"{where} page_image_id {p['page_image_id'][:12]}… 不在 page_images 表"
            )
        if p["page_image_id"] and p["render_dpi"] is None:
            report.error("ref", f"{where} 有 page_image_id 但 render_dpi 为空")
        block_ids = {b["image_id"] for b in (p["blocks"] or ()) if b["image_id"]}
        if not block_ids <= set(listed):
            report.error(
                "image_ids",
                f"{where} blocks 里引用了 image_ids 未列出的图: "
                f"{sorted(i[:12] for i in block_ids - set(listed))}",
            )


def _check_markers(
    pages: list[dict], images: list[dict], rasters: list[dict], report: Report
) -> None:
    """text 里的图标记必须能解析出去 —— 这是格式的核心承诺。"""
    have_img = {r["image_id"] for r in images}
    for p in pages:
        where = f"{p['doc_id'][:12]}…/p{p['page_index']}"
        for raw in IMAGE_REF_RE.findall(p["text"] or ""):
            ref = parse_image_ref(raw)
            if ref is None:
                report.error("marker", f"{where} 无法解析的图标记 {raw[:40]}")
            elif ref.kind == "blob":
                if ref.image_id not in have_img:
                    report.error("marker", f"{where} 标记指向不存在的图 {ref.image_id[:12]}…")
                elif ref.image_id not in (p["image_ids"] or ()):
                    report.error(
                        "marker", f"{where} 标记里的图未出现在 image_ids: {ref.image_id[:12]}…"
                    )
            elif ref.kind == "region":
                if not p["page_image_id"]:
                    report.error(
                        "marker",
                        f"{where} 用了 bbox:// 区域标记，但这一页没有整页光栅，"
                        f"图块取不出来",
                    )
                if any(v < 0.0 or v > 1.0 for v in ref.bbox or ()):
                    report.error("marker", f"{where} 区域标记 bbox 越界: {ref.bbox}")


def _check_media_tables(
    images: list[dict],
    rasters: list[dict],
    pages: list[dict],
    report: Report,
    verify_hashes: bool,
) -> None:
    if images and rasters:
        report.warn(
            "images-mode",
            f"images/ 和 page_images/ 同时有数据（{len(images)} / {len(rasters)}）。"
            f"MinerU 的裁剪图本就是整页光栅的子矩形，两个都存等于同一批像素存两遍",
        )

    referenced = {i for p in pages for i in (p["image_ids"] or ())}
    for table, name, keyed in ((images, "images", True), (rasters, "page_images", False)):
        seen = set()
        for r in table:
            iid = r["image_id"]
            if iid in seen:
                report.error(name, f"{name} 表里 image_id {iid[:12]}… 重复")
            seen.add(iid)
            if not _SHA256.match(iid or ""):
                report.error(name, f"{name} 的 image_id 不是 64 位小写十六进制: {iid!r}")
            blob = (r["image"] or {}).get("bytes")
            if not blob:
                report.error(name, f"{name} 的 {iid[:12]}… 没有字节")
                continue
            if r["n_bytes"] != len(blob):
                report.error(
                    name, f"{name} 的 {iid[:12]}… n_bytes={r['n_bytes']} 实为 {len(blob)}"
                )
            if r["width"] == 0 or r["height"] == 0:
                report.warn(name, f"{name} 的 {iid[:12]}… 宽高解析失败（format={r['format']}）")
            if verify_hashes:
                actual = hashlib.sha256(blob).hexdigest()
                if actual != iid:
                    report.error(
                        name,
                        f"{name} 的 {iid[:12]}… 内容寻址对不上，实际 sha256 是 {actual[:12]}…",
                    )
        if keyed:
            for iid in seen - referenced:
                report.warn("orphan", f"images 表里的 {iid[:12]}… 没有任何页引用")

    raster_keys = Counter((r["doc_id"], r["page_index"], r["render_dpi"]) for r in rasters)
    for key, n in raster_keys.items():
        if n > 1:
            report.error(
                "page_images", f"{key[0][:12]}… 第 {key[1]} 页 @{key[2]}dpi 有 {n} 张光栅"
            )


def _check_pairs(
    pairs: list[dict], pages: list[dict], images: list[dict], report: Report
) -> None:
    if not pairs:
        return
    page_keys = {(p["doc_id"], p["page_index"]) for p in pages}
    have_img = {r["image_id"] for r in images}
    for r in pairs:
        where = f"{r['doc_id'][:12]}…/p{r['page_index']}"
        if (r["doc_id"], r["page_index"]) not in page_keys:
            report.error("pairs", f"{where} 指向不存在的页")
        has_blob, has_bbox = bool(r["image_id"]), r["bbox"] is not None
        if has_blob == has_bbox:
            report.error(
                "pairs",
                f"{where} image_id 与 bbox 必须恰好有一个非空（现在 "
                f"{'两个都有' if has_blob else '两个都空'}）",
            )
        if has_blob and r["image_id"] not in have_img:
            report.error("pairs", f"{where} image_id {r['image_id'][:12]}… 不在 images 表")
        if not (r["text"] or "").strip():
            report.error("pairs", f"{where} 配对文本为空")


def _collect_stats(
    pages: list[dict],
    images: list[dict],
    rasters: list[dict],
    pairs: list[dict],
    report: Report,
) -> None:
    docs = {p["doc_id"] for p in pages}
    report.stats.update(
        {
            "documents": len(docs),
            "pages": len(pages),
            "images": len(images),
            "page_images": len(rasters),
            "pairs": len(pairs),
            "empty_pages": sum(1 for p in pages if not (p["text"] or "").strip()),
            "pages_without_blocks": sum(1 for p in pages if p["blocks"] is None),
            "total_chars": sum(p["n_chars"] or 0 for p in pages),
            "replacement_chars": sum((p["text"] or "").count("�") for p in pages),
            "by_extractor": dict(Counter(p["extractor"] for p in pages)),
            "block_types": dict(
                Counter(b["type"] for p in pages for b in (p["blocks"] or ()))
            ),
            "pair_sources": dict(Counter(r["source"] for r in pairs)),
        }
    )


def _read(directory: Path) -> list[dict]:
    if not directory.is_dir():
        return []
    rows: list[dict] = []
    for path in sorted(directory.glob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return rows
