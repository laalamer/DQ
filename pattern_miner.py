import pandas as pd
import numpy as np
from itertools import combinations
from difflib import SequenceMatcher


# ==================================================
# Helper Functions
# ==================================================

def is_numeric(series):
    return pd.api.types.is_numeric_dtype(series)


def is_datetime_like(series, threshold=0.8):
    converted = pd.to_datetime(series, errors="coerce")
    return converted.notna().mean() >= threshold


def is_categorical(series, max_unique_ratio=0.3, max_unique_values=50):
    unique_count = series.nunique(dropna=True)
    total = len(series)

    if total == 0:
        return False

    unique_ratio = unique_count / total

    return (
        unique_ratio <= max_unique_ratio
        and unique_count <= max_unique_values
    )


def normalize_text(value):
    if pd.isna(value):
        return None

    return str(value).strip().lower()


def text_similarity(a, b):
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


# ==================================================
# 1. Uniqueness Patterns
# ==================================================

def detect_uniqueness_patterns(df, threshold=0.95):
    patterns = []
    total_rows = len(df)

    if total_rows == 0:
        return patterns

    for col in df.columns:
        unique_count = df[col].nunique(dropna=True)
        uniqueness_ratio = unique_count / total_rows

        if uniqueness_ratio >= threshold:
            patterns.append({
                "pattern_type": "uniqueness",
                "column": col,
                "unique_count": int(unique_count),
                "total_rows": int(total_rows),
                "uniqueness_ratio": round(uniqueness_ratio * 100, 2),
                "evidence": (
                    f"Column '{col}' has {unique_count} unique values "
                    f"out of {total_rows} rows."
                )
            })

    return patterns


# ==================================================
# 2. Functional Dependencies
# Example: EmployeeID -> EmployeeName
# ==================================================

def detect_functional_dependencies(df, threshold=0.95, max_unique_values=500):
    patterns = []
    columns = list(df.columns)

    for determinant, dependent in combinations(columns, 2):
        if determinant == dependent:
            continue

        if df[determinant].nunique(dropna=True) > max_unique_values:
            continue

        temp = df[[determinant, dependent]].dropna()

        if temp.empty:
            continue

        grouped = temp.groupby(determinant)[dependent].nunique(dropna=True)

        if len(grouped) == 0:
            continue

        valid_groups = (grouped <= 1).sum()
        confidence = valid_groups / len(grouped)

        if confidence >= threshold:
            patterns.append({
                "pattern_type": "functional_dependency",
                "determinant_column": determinant,
                "dependent_column": dependent,
                "confidence": round(confidence * 100, 2),
                "evidence": (
                    f"For most values of '{determinant}', there is only one "
                    f"corresponding value in '{dependent}'."
                )
            })

    return patterns


# ==================================================
# 3. Conditional Dependencies
# Example: Status = Active -> ExitDate is empty
# ==================================================

def detect_conditional_dependencies(
    df,
    min_group_size=10,
    confidence_threshold=0.9,
    max_unique_values=30
):
    patterns = []

    categorical_cols = [
        col for col in df.columns
        if is_categorical(df[col], max_unique_values=max_unique_values)
    ]

    for condition_col in categorical_cols:
        condition_values = df[condition_col].dropna().unique()

        for condition_value in condition_values:
            subset = df[df[condition_col] == condition_value]

            if len(subset) < min_group_size:
                continue

            for target_col in df.columns:
                if target_col == condition_col:
                    continue

                missing_ratio = subset[target_col].isna().mean()

                if missing_ratio >= confidence_threshold:
                    patterns.append({
                        "pattern_type": "conditional_missing_dependency",
                        "condition_column": condition_col,
                        "condition_value": str(condition_value),
                        "target_column": target_col,
                        "expected_behavior": "missing",
                        "confidence": round(missing_ratio * 100, 2),
                        "support_count": int(len(subset)),
                        "evidence": (
                            f"When '{condition_col}' = '{condition_value}', "
                            f"'{target_col}' is missing in "
                            f"{round(missing_ratio * 100, 2)}% of rows."
                        )
                    })

                if is_categorical(df[target_col], max_unique_values=max_unique_values):
                    value_counts = subset[target_col].dropna().value_counts()

                    if value_counts.empty:
                        continue

                    top_value = value_counts.index[0]
                    top_count = value_counts.iloc[0]
                    confidence = top_count / len(subset)

                    if confidence >= confidence_threshold:
                        patterns.append({
                            "pattern_type": "conditional_value_dependency",
                            "condition_column": condition_col,
                            "condition_value": str(condition_value),
                            "target_column": target_col,
                            "expected_value": str(top_value),
                            "confidence": round(confidence * 100, 2),
                            "support_count": int(len(subset)),
                            "evidence": (
                                f"When '{condition_col}' = '{condition_value}', "
                                f"'{target_col}' is usually '{top_value}' "
                                f"in {round(confidence * 100, 2)}% of rows."
                            )
                        })

    return patterns


# ==================================================
# 4. Missing Value Patterns
# ==================================================

def detect_missing_patterns(df):
    patterns = []
    total_rows = len(df)

    if total_rows == 0:
        return patterns

    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_percent = missing_count / total_rows

        if missing_count > 0:
            patterns.append({
                "pattern_type": "missing_values",
                "column": col,
                "missing_count": int(missing_count),
                "missing_percent": round(missing_percent * 100, 2),
                "evidence": (
                    f"Column '{col}' has {missing_count} missing values "
                    f"out of {total_rows} rows."
                )
            })

    return patterns


# ==================================================
# 5. Rare Values
# Example: Mle, Femeate
# ==================================================

def detect_rare_values(df, rare_threshold=0.01, max_unique_values=50):
    patterns = []

    for col in df.columns:
        if is_numeric(df[col]):
            continue

        if df[col].nunique(dropna=True) > max_unique_values:
            continue

        value_counts = df[col].dropna().astype(str).value_counts()
        total = value_counts.sum()

        if total == 0:
            continue

        for value, count in value_counts.items():
            percentage = count / total

            if percentage <= rare_threshold:
                patterns.append({
                    "pattern_type": "rare_value",
                    "column": col,
                    "value": str(value),
                    "frequency": int(count),
                    "percentage": round(percentage * 100, 3),
                    "evidence": (
                        f"Value '{value}' appears rarely in column '{col}' "
                        f"with frequency {count}."
                    )
                })

    return patterns


# ==================================================
# 6. Fuzzy Standardization Issues
# Example: Mle ≈ Male, Femeate ≈ Female
# ==================================================

def detect_fuzzy_standardization_issues(
    df,
    similarity_threshold=0.75,
    rare_threshold=0.05,
    max_unique_values=50
):
    patterns = []

    for col in df.columns:
        if is_numeric(df[col]):
            continue

        unique_values = df[col].dropna().astype(str).unique()

        if len(unique_values) > max_unique_values:
            continue

        value_counts = df[col].dropna().astype(str).value_counts()
        total = value_counts.sum()

        if total == 0:
            continue

        common_values = [
            value for value, count in value_counts.items()
            if (count / total) >= rare_threshold
        ]

        rare_values = [
            value for value, count in value_counts.items()
            if (count / total) < rare_threshold
        ]

        for rare_value in rare_values:
            best_match = None
            best_score = 0

            for common_value in common_values:
                score = text_similarity(rare_value, common_value)

                if score > best_score:
                    best_score = score
                    best_match = common_value

            if best_match and best_score >= similarity_threshold:
                patterns.append({
                    "pattern_type": "fuzzy_standardization_issue",
                    "column": col,
                    "rare_value": str(rare_value),
                    "suggested_standard_value": str(best_match),
                    "similarity_score": round(best_score * 100, 2),
                    "evidence": (
                        f"Value '{rare_value}' in column '{col}' is similar "
                        f"to common value '{best_match}'."
                    )
                })

    return patterns


# ==================================================
# 7. Numeric Outliers
# ==================================================

def detect_numeric_outliers(df):
    patterns = []

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        series = df[col].dropna()

        if len(series) < 5:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = outlier_mask.sum()

        if outlier_count > 0:
            patterns.append({
                "pattern_type": "numeric_outliers",
                "column": col,
                "outlier_count": int(outlier_count),
                "lower_bound": round(float(lower_bound), 2),
                "upper_bound": round(float(upper_bound), 2),
                "evidence": (
                    f"Column '{col}' has {outlier_count} numeric outliers "
                    f"based on the IQR method."
                )
            })

    return patterns


# ==================================================
# 8. Date Relationships
# Example: EndDate should be after StartDate
# ==================================================

def detect_date_relationships(df, confidence_threshold=0.95):
    patterns = []
    date_cols = []

    for col in df.columns:
        if is_datetime_like(df[col]):
            date_cols.append(col)

    for col_a, col_b in combinations(date_cols, 2):
        date_a = pd.to_datetime(df[col_a], errors="coerce")
        date_b = pd.to_datetime(df[col_b], errors="coerce")

        valid = date_a.notna() & date_b.notna()

        if valid.sum() == 0:
            continue

        ratio_b_after_a = (date_b[valid] >= date_a[valid]).mean()
        ratio_a_after_b = (date_a[valid] >= date_b[valid]).mean()

        if ratio_b_after_a >= confidence_threshold:
            patterns.append({
                "pattern_type": "date_relationship",
                "first_date_column": col_a,
                "second_date_column": col_b,
                "expected_order": f"{col_b} >= {col_a}",
                "confidence": round(ratio_b_after_a * 100, 2),
                "support_count": int(valid.sum()),
                "evidence": (
                    f"In most valid rows, '{col_b}' is greater than or equal "
                    f"to '{col_a}'."
                )
            })

        if ratio_a_after_b >= confidence_threshold:
            patterns.append({
                "pattern_type": "date_relationship",
                "first_date_column": col_b,
                "second_date_column": col_a,
                "expected_order": f"{col_a} >= {col_b}",
                "confidence": round(ratio_a_after_b * 100, 2),
                "support_count": int(valid.sum()),
                "evidence": (
                    f"In most valid rows, '{col_a}' is greater than or equal "
                    f"to '{col_b}'."
                )
            })

    return patterns


# ==================================================
# 9. Main Function
# ==================================================

def discover_patterns(df):
    all_patterns = []

    all_patterns.extend(detect_uniqueness_patterns(df))
    all_patterns.extend(detect_functional_dependencies(df))
    all_patterns.extend(detect_conditional_dependencies(df))
    all_patterns.extend(detect_missing_patterns(df))
    all_patterns.extend(detect_rare_values(df))
    all_patterns.extend(detect_fuzzy_standardization_issues(df))
    all_patterns.extend(detect_numeric_outliers(df))
    all_patterns.extend(detect_date_relationships(df))

    return all_patterns