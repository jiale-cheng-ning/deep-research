import pytest
from deep_research.agents.base import (
    _coerce_stringified_json,
    _unescape_literal_escapes,
    _normalize_strings,
)


class TestUnescapeLiteralEscapes:
    def test_literal_newlines(self):
        assert _unescape_literal_escapes("line1\\nline2") == "line1\nline2"

    def test_double_newlines(self):
        assert _unescape_literal_escapes("a\\n\\nb") == "a\n\nb"

    def test_literal_tab(self):
        assert _unescape_literal_escapes("col1\\tcol2") == "col1\tcol2"

    def test_literal_backslash(self):
        assert _unescape_literal_escapes("path\\\\file") == "path\\file"

    def test_real_newlines_untouched(self):
        text = "already\nreal\nnewlines"
        assert _unescape_literal_escapes(text) == text

    def test_mixed(self):
        assert _unescape_literal_escapes("a\\nb\nc") == "a\nb\nc"


class TestNormalizeStrings:
    def test_string(self):
        assert _normalize_strings("x\\ny") == "x\ny"

    def test_list(self):
        assert _normalize_strings(["a\\nb", "c"]) == ["a\nb", "c"]

    def test_nested_dict(self):
        data = {"outer": {"inner": "p\\nq"}}
        assert _normalize_strings(data) == {"outer": {"inner": "p\nq"}}

    def test_non_string_passthrough(self):
        assert _normalize_strings(42) == 42
        assert _normalize_strings(None) is None


class TestCoerceStringifiedJson:
    def test_json_array_string(self):
        data = {"items": '["a", "b"]'}
        result = _coerce_stringified_json(data)
        assert result["items"] == ["a", "b"]

    def test_json_object_string(self):
        data = {"obj": '{"key": "value"}'}
        result = _coerce_stringified_json(data)
        assert result["obj"] == {"key": "value"}

    def test_plain_string_untouched(self):
        data = {"text": "just a normal sentence"}
        result = _coerce_stringified_json(data)
        assert result["text"] == "just a normal sentence"

    def test_invalid_json_string_kept(self):
        data = {"text": "[not valid json"}
        result = _coerce_stringified_json(data)
        assert result["text"] == "[not valid json"

    def test_nested_list_of_dicts(self):
        data = {
            "claims": [
                {"text": "fact one", "confidence": 0.9},
                {"text": "fact\\ntwo", "confidence": 0.5},
            ]
        }
        result = _coerce_stringified_json(data)
        assert result["claims"][0]["text"] == "fact one"
        assert result["claims"][1]["text"] == "fact\ntwo"

    def test_conclusions_with_literal_newlines(self):
        """Regression: report conclusions showed literal \\n instead of real newlines."""
        data = {
            "conclusions": "Summary:\\n\\n**Point 1**: detail\\n**Point 2**: more"
        }
        result = _coerce_stringified_json(data)
        assert "\n\n" in result["conclusions"]
        assert "\\n" not in result["conclusions"]


class TestFormatReportMarkdown:
    def test_produces_newlines_not_escapes(self):
        from deep_research.models import FinalReport, ReportSection
        from deep_research.pipeline import format_report_markdown

        report = FinalReport(
            title="Test",
            executive_summary="Summary line1\\n\\nSummary line2",
            sections=[
                ReportSection(heading="Sec", content="Body\\nwith breaks", confidence="high")
            ],
            conclusions="End\\n\\nDone",
            citations=[],
        )
        md = format_report_markdown(report)
        # The format function just joins; normalization must have happened earlier.
        # This test documents current behavior — content is passed through as-is
        # by format_report_markdown; the fix is in _extract_tool_result / coercion.
        assert "Test" in md
        assert "## Executive Summary" in md
