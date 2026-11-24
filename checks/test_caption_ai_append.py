"""Regression coverage for Auto-Captioner append and prompt forwarding."""
from fizgig.scripts.batch_caption import merge_caption, request_for
from fizgig import captioners


def test_append_tags_preserves_old_and_deduplicates():
    assert merge_caption("ohwx, blue eyes, cat", "ohwx, BLUE_EYES, dog, Dog", "tags", "ohwx") == "ohwx, blue eyes, cat, dog"
    assert merge_caption("cat", "cat", "tags") == "cat"
    assert merge_caption("", "ohwx, cat", "tags", "ohwx") == "ohwx, cat"


def test_append_natural_preserves_caption_and_trigger():
    assert merge_caption("ohwx, A cat sits.", "ohwx, The cat looks left.", "natural", "ohwx") == "ohwx, A cat sits.\nThe cat looks left."
    assert merge_caption("A cat sits.", "A cat sits.", "natural") == "A cat sits."


def test_request_carries_system_and_user_prompt():
    request, why = request_for(captioners.get("qwen3vl"), {"instruction": "Return tags.", "system_prompt": "You are a tagger."})
    assert not why
    assert request.instruction == "Return tags."
    assert request.system_prompt == "You are a tagger."


def test_append_both_directions_keeps_tag_line_first():
    assert merge_caption("cat, blue eyes", "A cat sits. It looks left.", "natural") == "cat, blue eyes\nA cat sits. It looks left."
    assert merge_caption("A cat sits. It looks left.", "cat, blue eyes", "tags") == "cat, blue eyes\nA cat sits. It looks left."
    assert merge_caption("cat, blue eyes\nA cat sits. It looks left.", "CAT, green eyes", "tags") == "cat, blue eyes, green eyes\nA cat sits. It looks left."
    assert merge_caption("ohwx, A cat sits.", "ohwx, cat, blue eyes", "tags", "ohwx") == "ohwx, cat, blue eyes\nA cat sits."
    assert merge_caption("cat; blue eyes\nA cat sits.", "CAT, dog", "tags", delimiter=";") == "cat; blue eyes; dog\nA cat sits."
