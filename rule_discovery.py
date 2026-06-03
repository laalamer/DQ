import json
import re

from src.profiler import build_dataset_profile
from src.pattern_miner import discover_patterns
from src.prompts import build_rule_discovery_prompt
from src.llama_client import ask_llama


def extract_json_from_text(text):
    """
    Extract JSON array from LLM response.
    """

    if not text:
        return []

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", text, re.DOTALL)

    if match:
        json_text = match.group(0)

        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            return []

    return []


def normalize_rules(rules):
    """
    Make sure rules have consistent IDs and required fields.
    """

    normalized = []

    for idx, rule in enumerate(rules, start=1):
        if not isinstance(rule, dict):
            continue

        normalized_rule = {
            "rule_id": rule.get("rule_id", f"DQ_{idx:02d}"),
            "rule_name": rule.get("rule_name", "Unnamed Rule"),
            "description": rule.get("description", ""),
            "rule_type": rule.get("rule_type", "unknown"),
            "source_pattern_type": rule.get("source_pattern_type", ""),
            "columns": rule.get("columns", []),
            "condition": rule.get("condition", {}),
            "expectation": rule.get("expectation", {}),
            "confidence": rule.get("confidence", None),
            "reason": rule.get("reason", "")
        }

        normalized.append(normalized_rule)

    return normalized


def discover_rules(df):
    """
    Full rule discovery pipeline:
    1. Build profile
    2. Discover patterns using Python
    3. Ask LLaMA to evaluate patterns
    4. Return approved DQ rules
    """

    profile = build_dataset_profile(df)
    patterns = discover_patterns(df)

    prompt = build_rule_discovery_prompt(
        profile=profile,
        patterns=patterns
    )

    llm_response = ask_llama(prompt)

    rules = extract_json_from_text(llm_response)
    rules = normalize_rules(rules)

    return {
        "profile": profile,
        "patterns": patterns,
        "llm_response": llm_response,
        "rules": rules
    }