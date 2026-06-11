"""Shared parsing utilities used by all VLM backends."""

import json
import re
from typing import Any, Dict

from json_repair import repair_json
from morphic import validate


@validate
def parse_json_from_text(raw_text: str) -> Dict[str, Any]:
    """Extract and parse the first JSON object from model output text.

    Strips markdown fences, fixes bare numeric keys, repairs common LLM
    JSON errors (invalid escapes, trailing commas, single-quoted keys,
    missing closing braces) via ``json_repair``, then parses with
    ``json.loads``.  Raises ``ValueError`` if no valid JSON is found.
    """
    cleaned: str = raw_text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "")

    start: int = cleaned.find("{")
    end: int = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON content found in model output.")

    json_str: str = cleaned[start : end + 1]
    json_str = re.sub(r"(\{|,)\s*(\d+)\s*:", r'\1 "\2":', json_str)

    repaired: str = repair_json(json_str)
    return json.loads(repaired)
