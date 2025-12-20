from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def _example_normal_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "city": ["Moscow", "Saint Petersburg", "Almaty", "Minsk", "Moscow", "Novosibirsk", "Kyiv", "Moscow", "Astana"],
        }
    )


def _example_duplicates_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 4, 4, 4, 8, 9],
            "city": ["Moscow", "Moscow", "Moscow", "Moscow", "Moscow", "Moscow", "Moscow", "Moscow", "Moscow"],
        }
    )


def _example_missing_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 4, 4, 4, 8, 9],
            "city": ["Moscow", "Moscow", "Moscow", "Moscow", "Moscow", "Moscow", "Moscow", "Moscow", "Moscow"],
            "corrupted": [1, 2, None, None, None, 100, 100, None, 101],
        }
    )


def test_compute_quality_flags():
    df_normal = _example_normal_df()
    summary_normal = summarize_dataset(df_normal)
    missing_df_normal = missing_table(df_normal)
    quality_flags_normal = compute_quality_flags(summary_normal, missing_df_normal)

    assert quality_flags_normal["has_constant_columns"] == False
    assert quality_flags_normal["has_suspicious_id_duplicates"] == False

    df_duplicates = _example_duplicates_df()
    summary_duplicates = summarize_dataset(df_duplicates)
    missing_df_duplicates = missing_table(df_duplicates)
    quality_flags_duplicates = compute_quality_flags(summary_duplicates, missing_df_duplicates)

    assert not quality_flags_duplicates["has_constant_columns"] == False
    assert not quality_flags_duplicates["has_suspicious_id_duplicates"] == False

    df_missing = _example_missing_df()
    summary_missing = summarize_dataset(df_missing)
    missing_df_missing = missing_table(df_missing)
    quality_flags_missing = compute_quality_flags(summary_missing, missing_df_missing)

    assert not quality_flags_missing["high_missing_columns"] == False


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2
