"""Strict single-document JSON parsing for CLI output with presentation noise."""

from __future__ import annotations

from npa.clients.json_output import parse_single_json_document


def test_parses_document_with_diagnostic_preamble() -> None:
    assert parse_single_json_document('note\n[{"a": 1}]') == [{"a": 1}]


def test_tolerates_trailing_ansi_spinner_frame() -> None:
    # SkyPilot's rich status spinner can flush one final frame to stdout after
    # the JSON payload; ANSI erase/cursor sequences contain ESC+"[" which must
    # not be mistaken for a second JSON document.
    output = (
        '[{"job_id": 5, "status": "RUNNING"}]\n'
        "\x1b[2K\x1b[32m⠏\x1b[0m \x1b[36mChecking managed jobs\x1b[0m\n"
        "\x1b[?25h\x1b[1A\x1b[2K"
    )
    assert parse_single_json_document(output) == [{"job_id": 5, "status": "RUNNING"}]


def test_rejects_second_document_after_payload() -> None:
    assert parse_single_json_document('{"a": 1}\n{"b": 2}') is None


def test_rejects_ambiguous_preamble_value() -> None:
    assert parse_single_json_document("junk [1, 2] trailing {") is None


def test_rejects_trailing_json_start_outside_ansi() -> None:
    assert parse_single_json_document('[{"a": 1}]\n[') is None


def test_tolerates_json_looking_fragment_quoted_in_a_warning() -> None:
    # `sky status --output json` can print this exact warning to stdout ahead of
    # the real payload when the client/server allowed_clouds config disagrees.
    # Its parenthesized aside `(["allowed_clouds"])` parses as a valid one-item
    # JSON array, but it is prose, not a second document.
    output = (
        'The following keys (["allowed_clouds"]) have different values in the '
        "client SkyPilot config with the server and will be ignored. Remove "
        "these keys to disable this warning. If you want to specify it, please "
        "modify it on server side or contact your administrator.\n"
        '[\n  {"name": "sky-jobs-controller-dd17c189", "status": "UP"}\n]\n'
    )
    assert parse_single_json_document(output) == [
        {"name": "sky-jobs-controller-dd17c189", "status": "UP"}
    ]


def test_still_rejects_ambiguous_value_flanked_by_whitespace() -> None:
    # Unlike a fragment embedded in punctuation, a JSON-looking value that is
    # itself whitespace-flanked is a real competing document and must still
    # reject via the existing trailing-content check.
    assert parse_single_json_document("junk [1, 2] trailing {") is None
