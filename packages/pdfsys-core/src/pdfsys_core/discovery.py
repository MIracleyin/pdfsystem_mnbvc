"""Finding the PDFs in a corpus, and reading a worklist of them.

Every entry point that takes ``--pdf-dir`` has to agree on what counts as a
PDF, or a shard silently covers a different corpus than the run it claims to
come from. So the rule lives here, once.

The rule is: a ``.pdf`` suffix in any case is taken at its word, and every
other file is judged by whether it begins with ``%PDF-``. The plain
``rglob("*.pdf")`` this replaces matched case-sensitively on Linux and macOS,
and never looked inside anything — so the loss was both large and invisible.

Sniffing everything else, rather than only extensionless files, is a decision
made against a real corpus. Measured on 218,297 files of cmn_Hani:

    .pdf by suffix     199,992
    by %PDF- header     18,005     ← 8.3% of the corpus
    genuinely not PDF      300

Two thirds of that 18,005 carry a suffix — ``.ashx``, ``.php``, ``.aspx``,
``.cgi``, ``.jsp`` — because a scraper saved them under the last path segment
of a URL like ``download.ashx?id=123``. Trusting those suffixes drops 4.4% of
the corpus; trusting only extensionless files still drops it. The scan costs
286 s cold on that corpus against 0.5 s for the walk alone, which is noise
beside the days of extraction it precedes, and a run that skips 18,000
documents is not cheaper — it is wrong.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

_LOG = logging.getLogger(__name__)

#: Every PDF starts with this. The spec allows leading junk before it, but a
#: file that needs that allowance is not one we want to find by guessing.
PDF_MAGIC = b"%PDF-"

__all__ = [
    "PDF_MAGIC",
    "PdfInventory",
    "Worklist",
    "iter_pdf_paths",
    "looks_like_pdf",
    "read_pdf_list",
    "take_inventory",
]


@dataclass(frozen=True, slots=True)
class PdfInventory:
    """What a scan found, split by how it recognised each file."""

    by_suffix: tuple[Path, ...] = ()
    by_magic: tuple[Path, ...] = ()
    #: Directories the scan could not enter. Their contents are missing from
    #: this inventory, and a corpus is not smaller just because part of it was
    #: unreadable — so the caller has to be able to say so.
    unreadable_dirs: tuple[str, ...] = ()

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(sorted([*self.by_suffix, *self.by_magic]))

    def __len__(self) -> int:
        return len(self.by_suffix) + len(self.by_magic)

    def describe(self) -> str:
        parts = [f"{len(self)} PDFs"]
        if self.by_magic:
            parts.append(
                f"{len(self.by_suffix)} 个按后缀，"
                f"{len(self.by_magic)} 个按 %PDF- 文件头识别"
            )
        if self.unreadable_dirs:
            parts.append(f"{len(self.unreadable_dirs)} 个目录读不进去")
        return parts[0] + ("（" + "，".join(parts[1:]) + "）" if len(parts) > 1 else "")


def looks_like_pdf(path: str | Path) -> bool:
    """Apply the module's rule to one path already in hand.

    The scan decides what to yield; this decides whether something a caller was
    handed — a worklist entry — qualifies. Both have to be the same rule, or a
    list-driven shard covers a different corpus than a scan-driven one.
    """
    path = Path(path)
    if not path.is_file():
        return False
    if os.path.splitext(path.name)[1].strip().lower() == ".pdf":
        return True
    return _looks_like_pdf(path)


def _looks_like_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(len(PDF_MAGIC)) == PDF_MAGIC
    except OSError as e:
        _LOG.warning("cannot read %s while looking for PDFs: %s", path, e)
        return False


def take_inventory(root: str | Path, *, sniff: bool = True) -> PdfInventory:
    """Scan *root* recursively and report what was found, and how.

    Walks with :func:`os.walk` rather than ``rglob("*")`` because rglob swallows
    the ``PermissionError`` from descending into an unreadable directory and
    simply yields nothing, which turns "1000 documents you cannot read" into
    "1000 documents that do not exist".

    ``sniff=False`` trusts suffixes and reads nothing. It is much faster on a
    tree that is mostly not PDFs, and on a scraped corpus it silently drops
    every document whose URL ended in ``.ashx`` — see the module docstring for
    what that cost on a real one.
    """
    by_suffix: list[Path] = []
    by_magic: list[Path] = []
    unreadable: list[str] = []

    def _on_error(exc: OSError) -> None:
        _LOG.warning("cannot descend into %s: %s", exc.filename, exc)
        unreadable.append(str(exc.filename))

    for dirpath, _dirnames, filenames in os.walk(root, onerror=_on_error):
        for name in filenames:
            # splitext, not Path.suffix, to avoid allocating a Path for every
            # entry in the tree. Stripped because a suffix carrying stray
            # whitespace ("scan.PDF ", "x.pdf\r" from a CRLF-mangled unzip) is
            # neither a clean ".pdf" nor absent, and fell between both branches.
            suffix = os.path.splitext(name)[1].strip()
            path = Path(dirpath) / name
            if suffix.lower() == ".pdf":
                if path.is_file():
                    by_suffix.append(path)
            elif sniff and path.is_file() and _looks_like_pdf(path):
                by_magic.append(path)

    return PdfInventory(
        tuple(sorted(by_suffix)), tuple(sorted(by_magic)), tuple(sorted(unreadable))
    )


def iter_pdf_paths(root: str | Path, *, sniff: bool = True) -> Iterator[Path]:
    """Yield every PDF under *root*, sorted. See the module docstring for what counts."""
    yield from take_inventory(root, sniff=sniff).paths


@dataclass(frozen=True, slots=True)
class Worklist:
    """A parsed worklist, and everything it did not turn into work."""

    paths: tuple[Path, ...] = ()
    missing: tuple[str, ...] = ()
    duplicates: tuple[str, ...] = ()
    entries: int = 0

    def describe(self) -> str:
        parts = [f"{len(self.paths)}/{self.entries} 条可用"]
        if self.missing:
            parts.append(f"{len(self.missing)} 条文件不存在")
        if self.duplicates:
            parts.append(f"{len(self.duplicates)} 条重复")
        return "，".join(parts)


def read_pdf_list(
    list_path: str | Path, *, path_root: str | Path | None = None
) -> Worklist:
    """Read a worklist of PDF paths, one per line.

    Every entry ends up in exactly one of ``paths``, ``missing`` or
    ``duplicates``, and ``entries`` is their total — so a caller can always say
    what became of the file it was handed, rather than quietly processing a
    shorter corpus than it was given.

    ``path_root`` re-anchors relative entries. That is what makes a worklist
    portable: the CPU box writes paths relative to its corpus root, the GPU box
    reads the same file against wherever it put them. Absolute entries are
    taken as-is, because a caller that wrote an absolute path meant it.

    Order is preserved. A worklist is often a deliberate slice — ``split -n``
    over a bucket file — and re-sorting it would silently regroup the work.
    """
    root = Path(path_root) if path_root is not None else None
    paths: list[Path] = []
    missing: list[str] = []
    duplicates: list[str] = []
    entries = 0
    seen: set[Path] = set()

    # utf-8-sig drops a BOM if the list came out of a Windows editor, where it
    # would otherwise be glued to the first path and cost that one document.
    # surrogateescape because filenames are bytes: a corpus with one path the
    # locale cannot decode must lose that path, not the whole run.
    with Path(list_path).open(
        encoding="utf-8-sig", errors="surrogateescape"
    ) as f:
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            entries += 1
            path = Path(line)
            if root is not None and not path.is_absolute():
                path = root / path
            if not path.is_file():
                missing.append(line)
                continue
            resolved = path.resolve()
            if resolved in seen:
                # The same PDF twice would be extracted twice and, downstream,
                # violate the (doc_id, page_index) primary key.
                duplicates.append(line)
                continue
            seen.add(resolved)
            paths.append(path)
    return Worklist(tuple(paths), tuple(missing), tuple(duplicates), entries)
