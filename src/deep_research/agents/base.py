import json
import logging
import re
from abc import ABC, abstractmethod

from ..config import Settings
from ..llm import LLMClient, TokenUsage

logger = logging.getLogger("deep_research.agents")

_LITERAL_ESCAPES = re.compile(r"\\([nrt\\])")


def _unescape_literal_escapes(text: str) -> str:
    """Turn literal \\n / \\r / \\t / \\\\ sequences into real characters."""
    mapping = {"n": "\n", "r": "\r", "t": "\t", "\\": "\\"}
    return _LITERAL_ESCAPES.sub(lambda m: mapping[m.group(1)], text)


def _normalize_strings(data):
    """Recursively un-escape literal newlines/tabs in every string value."""
    if isinstance(data, str):
        return _unescape_literal_escapes(data)
    if isinstance(data, dict):
        return {k: _normalize_strings(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_normalize_strings(item) for item in data]
    return data


def _coerce_stringified_json(data: dict) -> dict:
    """Recursively convert string-serialized JSON arrays/objects back to Python types."""
    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            stripped = value.strip()
            if (stripped.startswith("[") and stripped.endswith("]")) or \
               (stripped.startswith("{") and stripped.endswith("}")):
                try:
                    result[key] = json.loads(stripped)
                except json.JSONDecodeError:
                    result[key] = value
            else:
                result[key] = value
        elif isinstance(value, dict):
            result[key] = _coerce_stringified_json(value)
        elif isinstance(value, list):
            result[key] = [
                _coerce_stringified_json(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value
    return _normalize_strings(result)


class BaseAgent(ABC):
    model_config_key: str = "model_sonnet"

    def __init__(self, llm: LLMClient, settings: Settings | None = None):
        self.llm = llm
        self.settings = settings
        self.name = self.__class__.__name__

    @property
    def model(self) -> str:
        if self.settings:
            return getattr(self.settings, self.model_config_key)
        return "claude-sonnet-4-6"

    @abstractmethod
    def system_prompt(self) -> str: ...

    @abstractmethod
    def output_schema(self) -> dict: ...

    def _make_structured_tool(self) -> dict:
        return {
            "name": "submit_result",
            "description": "Submit your structured analysis result. You MUST call this tool.",
            "input_schema": self.output_schema(),
        }

    def _extract_tool_result(self, messages: list[dict], model_class):
        # First: look for tool_use blocks
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                for block in msg.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        raw = block["input"]
                        data = _coerce_stringified_json(raw)
                        # Debug: log any fields still stringified
                        for k, v in data.items():
                            if isinstance(v, str) and v.strip().startswith(("[", "{")):
                                logger.warning("Field '%s' still string after coercion, retrying parse", k)
                                try:
                                    data[k] = json.loads(v)
                                except json.JSONDecodeError:
                                    pass
                        return model_class(**data)

        # Fallback: try to parse text content as JSON
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                for block in msg.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block["text"]
                        try:
                            # Use JSONDecoder to parse just the first JSON object
                            start = text.find("{")
                            if start >= 0:
                                decoder = json.JSONDecoder()
                                data, _ = decoder.raw_decode(text, start)
                                if isinstance(data, dict):
                                    data = _coerce_stringified_json(data)
                                    return model_class(**data)
                        except (json.JSONDecodeError, ValueError, Exception) as e:
                            logger.warning("Failed to parse text as JSON: %s", e)

        raise ValueError(f"No tool_use or parseable JSON found for {model_class.__name__}")
