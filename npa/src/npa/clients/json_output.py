"""Strict parsing for CLI JSON with a bounded diagnostic preamble."""

from __future__ import annotations

import json
import re
from typing import Any

# ANSI CSI/OSC control sequences (colors, cursor movement, erase-line) emitted
# by rich status spinners; their introducer is ESC+"[", which must not be
# mistaken for the start of a JSON array.
_ANSI_SEQUENCE_RE = re.compile(
    r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?"
)


def parse_single_json_document(output: str) -> Any | None:
    """Return one trailing JSON document, rejecting ambiguity and trailing text.

    Trailing output that cannot begin another JSON value (for example SkyPilot's
    rich status spinner occasionally flushing one final ``⠏ Checking managed
    jobs`` frame with ANSI control sequences after the JSON array) is not
    ambiguity and is ignored; any trailing ``[`` or ``{`` still rejects.

    A candidate span embedded inside ordinary prose (for example SkyPilot's own
    warning ``The following keys (["allowed_clouds"]) have different values...``,
    whose parenthesized aside happens to parse as the one-item JSON array
    ``["allowed_clouds"]``) is not a second document either: :func:`_is_standalone_span`
    requires whitespace/string-boundary flanking so quoted fragments like that
    one do not make the real trailing payload look ambiguous.
    """

    text = str(output or "")
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            payload, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if not _is_standalone_span(text, index, end):
            continue
        trailing = _ANSI_SEQUENCE_RE.sub("", text[end:])
        if any(ch in "[{" for ch in trailing) or _contains_json_value(
            text[:index], decoder
        ):
            continue
        return payload
    return None


def _is_standalone_span(text: str, start: int, end: int) -> bool:
    """True unless prose immediately flanks ``text[start:end]`` on either side.

    A tool's genuine JSON output is its own block: only whitespace (or the
    start/end of the string) surrounds it. A JSON-looking fragment quoted
    inside a sentence is instead flanked by ordinary punctuation or letters.
    """

    before = text[:start]
    if before and not before[-1].isspace():
        return False
    after = _ANSI_SEQUENCE_RE.sub("", text[end:])
    if after and not after[0].isspace():
        return False
    return True


def _contains_json_value(prefix: str, decoder: json.JSONDecoder) -> bool:
    for index, character in enumerate(prefix):
        if character not in "[{":
            continue
        try:
            _payload, end = decoder.raw_decode(prefix, index)
        except json.JSONDecodeError:
            continue
        if not _is_standalone_span(prefix, index, end):
            continue
        return True
    return False
