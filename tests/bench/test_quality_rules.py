"""Layer-1 hard-rule checks for extracted text quality.

These tests pin the contract for ``quality_rules.check_extracted_text`` —
a pure-stdlib, layout-free gate that runs BEFORE any model-based scorer.
The goal is to cheaply reject obviously-broken parser output (empty,
near-empty, garbage, repetition loops) so the cascade can escalate to
the next parser without paying for a BERT/LLM score on text we already
know is unusable.
"""

from __future__ import annotations

from pdfsys_bench.quality_rules import check_extracted_text


def test_empty_text_triggers_empty_output_blocker():
    result = check_extracted_text("", page_count=1)
    assert result.blockers["empty_output"] is True
    assert result.any_blocker is True


def test_whitespace_only_text_triggers_empty_output_blocker():
    result = check_extracted_text("   \n\t\n  ", page_count=3)
    assert result.blockers["empty_output"] is True


def test_clean_english_text_has_no_blockers():
    text = "This is a clean paragraph of extracted text. " * 20
    result = check_extracted_text(text, page_count=1)
    assert result.any_blocker is False, result.blockers


def test_clean_chinese_text_is_not_flagged_as_garbage():
    # Project is Chinese-heavy — CJK must never be mistaken for garbage.
    text = "这是一段正常的中文文本,应当被视为干净的提取结果。" * 5
    result = check_extracted_text(text, page_count=1)
    assert result.any_blocker is False, result.blockers


def test_too_short_for_page_count_triggers_too_short_blocker():
    # 100 chars across 50 pages = 2 chars/page, well below the default
    # min_chars_per_page=50. Catches "mupdf returned almost nothing
    # from a scanned PDF".
    text = "x" * 100
    result = check_extracted_text(text, page_count=50)
    assert result.blockers["too_short"] is True


def test_text_long_enough_per_page_does_not_trigger_too_short():
    # 50 pages of ~130 chars of distinct content — well above the
    # min_chars_per_page threshold. Lines must be distinct so the
    # repetition-loop check stays quiet.
    pages = [
        f"Page {i} content with enough characters to clear the threshold "
        f"and a unique marker word {word} to avoid repetition flagging."
        for i, word in enumerate(["alpha", "beta", "gamma", "delta", "epsilon"] * 10)
    ]
    text = "\n".join(pages)
    result = check_extracted_text(text, page_count=50)
    assert result.blockers["too_short"] is False


def test_high_replacement_char_ratio_triggers_replacement_blocker():
    # U+FFFD is the canonical "I gave up decoding this byte" marker.
    # Even 1% of these usually means encoding went wrong somewhere.
    text = ("normal text " * 10) + ("�" * 20)
    result = check_extracted_text(text, page_count=1)
    assert result.blockers["high_replacement_chars"] is True


def test_high_garbage_char_ratio_triggers_garbage_blocker():
    # Control / unassigned / private-use chars = encoding artefact.
    text = "hello world " + ("\x00\x01\x02\x03\x04\x05" * 30)
    result = check_extracted_text(text, page_count=1)
    assert result.blockers["high_garbage_chars"] is True


def test_repeating_line_loop_triggers_repetition_blocker():
    # Classic RolmOCR/VLM hallucination: same line emitted 20+ times.
    text = ("the model was unable to read this page\n" * 20) + "real content here"
    result = check_extracted_text(text, page_count=1)
    assert result.blockers["repetition_loop"] is True


def test_normal_text_with_some_duplicate_lines_does_not_trigger_repetition():
    # Boilerplate (page header, "References", etc.) can repeat a few times
    # in a legitimate document — only LONG runs should fire the blocker.
    text = (
        "Chapter 1: Introduction\n"
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.\n"
        "Section 1.1\n"
        "More body text continues here for a while.\n"
        "Chapter 2: Methods\n"
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.\n"
        "Section 2.1\n"
        "Another paragraph of distinct content.\n"
    ) * 3
    result = check_extracted_text(text, page_count=1)
    assert result.blockers["repetition_loop"] is False


def test_metrics_are_exposed_for_observability():
    text = "hello world " * 10
    result = check_extracted_text(text, page_count=1)
    # Raw measurements let callers log / debug why a blocker did/didn't fire.
    assert "chars_per_page" in result.metrics
    assert "replacement_ratio" in result.metrics
    assert "garbage_ratio" in result.metrics
    assert "max_repetition_run" in result.metrics


def test_multiple_blockers_can_fire_simultaneously():
    # A single U+FFFD across 100 pages: empty-ish AND replacement-heavy
    # AND too-short.
    result = check_extracted_text("�", page_count=100)
    assert result.blockers["high_replacement_chars"] is True
    assert result.blockers["too_short"] is True


def test_any_blocker_is_false_only_when_all_blockers_false():
    clean = check_extracted_text(
        "Just a clean sentence with more than fifty characters of body content here.",
        page_count=1,
    )
    assert clean.any_blocker is False, clean.blockers

    dirty = check_extracted_text("", page_count=1)
    assert dirty.any_blocker is True
