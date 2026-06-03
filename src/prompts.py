import json


def build_rule_discovery_prompt(profile, patterns):
    prompt = f"""
You are an AI Data Quality Analyst.

Your task is to evaluate detected data patterns and decide which ones represent valid Data Quality rules.

Important:
- Do NOT invent rules.
- Use only the provided dataset profile and detected patterns.
- Reject weak, meaningless, or coincidental patterns.
- Return only rules that are useful for data quality validation.
- The rules must be domain-agnostic and based on evidence.
- Do not return Python code.
- Return valid JSON only.

Dataset Profile:
{json.dumps(profile, indent=2, ensure_ascii=False)}

Detected Patterns:
{json.dumps(patterns, indent=2, ensure_ascii=False)}

Return the result in this JSON format:

[
  {{
    "rule_id": "DQ_01",
    "rule_name": "Short rule name",
    "description": "Clear business-readable data quality rule",
    "rule_type": "uniqueness | completeness | consistency | validity | standardization | date_logic | outlier",
    "source_pattern_type": "pattern type from detected patterns",
    "columns": ["column1", "column2"],
    "condition": {{
      "column": "condition column if exists",
      "operator": "=",
      "value": "condition value if exists"
    }},
    "expectation": {{
      "column": "target column",
      "operator": "is_null | is_not_null | equals | unique | greater_or_equal | in_standard_values | within_range",
      "value": "expected value if exists"
    }},
    "confidence": 95.0,
    "reason": "Why this pattern should be considered a data quality rule"
  }}
]

If no strong rules are found, return:
[]
"""
    return prompt
