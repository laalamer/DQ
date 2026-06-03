import pandas as pd


def validate_uniqueness(df, rule):

    column = rule["columns"][0]

    violations = df[
        df[column].duplicated(keep=False)
    ]

    return violations


def validate_is_null(df, rule):

    condition = rule["condition"]
    expectation = rule["expectation"]

    condition_column = condition["column"]
    condition_value = condition["value"]

    target_column = expectation["column"]

    violations = df[
        (df[condition_column] == condition_value)
        &
        (df[target_column].notna())
    ]

    return violations


def validate_is_not_null(df, rule):

    condition = rule["condition"]
    expectation = rule["expectation"]

    condition_column = condition["column"]
    condition_value = condition["value"]

    target_column = expectation["column"]

    violations = df[
        (df[condition_column] == condition_value)
        &
        (df[target_column].isna())
    ]

    return violations


def validate_equals(df, rule):

    condition = rule["condition"]
    expectation = rule["expectation"]

    condition_column = condition["column"]
    condition_value = condition["value"]

    target_column = expectation["column"]
    expected_value = expectation["value"]

    violations = df[
        (df[condition_column] == condition_value)
        &
        (df[target_column] != expected_value)
    ]

    return violations


def validate_date_relationship(df, rule):

    columns = rule["columns"]

    if len(columns) < 2:
        return pd.DataFrame()

    col_a = columns[0]
    col_b = columns[1]

    violations = df[
        pd.to_datetime(
            df[col_b],
            errors="coerce"
        )
        <
        pd.to_datetime(
            df[col_a],
            errors="coerce"
        )
    ]

    return violations


def validate_rule(df, rule):

    operator = (
        rule.get("expectation", {})
        .get("operator")
    )

    if operator == "unique":
        return validate_uniqueness(df, rule)

    elif operator == "is_null":
        return validate_is_null(df, rule)

    elif operator == "is_not_null":
        return validate_is_not_null(df, rule)

    elif operator == "equals":
        return validate_equals(df, rule)

    elif operator == "greater_or_equal":
        return validate_date_relationship(df, rule)

    return pd.DataFrame()
