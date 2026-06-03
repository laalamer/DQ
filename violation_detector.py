from src.rule_validator import validate_rule


def detect_violations(df, rules):

    results = []

    for rule in rules:

        violations = validate_rule(
            df,
            rule
        )

        results.append({

            "rule_id":
            rule["rule_id"],

            "rule_name":
            rule["rule_name"],

            "description":
            rule["description"],

            "violation_count":
            len(violations),

            "violating_rows":
            violations

        })

    return results