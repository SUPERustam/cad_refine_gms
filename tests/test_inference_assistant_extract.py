"""Regression tests for Qwen chat assistant span extraction used in inference."""

from cad_rl.pipelines.inference import _extract_assistant_text


def test_extract_assistant_text_strips_markers():
    decoded = (
        "<|im_start|>user\nhello<|im_end|>"
        "<|im_start|>assistant\nimport cadquery as cq\nr = cq.Workplane('XY').circle(1).extrude(1)<|im_end|>"
    )
    out = _extract_assistant_text(decoded)
    assert "import cadquery" in out
    assert "circle(1)" in out
    assert "im_start" not in out
    assert "<|im_end|>" not in out


def test_extract_assistant_text_fallback_full_decode():
    decoded = "no chat markers here"
    assert _extract_assistant_text(decoded) == "no chat markers here"
