from pathlib import Path

import numpy as np
import pandas as pd

def build_empirical_envelope(
    surrogates_df,
    lower_quantile=0.025,
    upper_quantile=0.975,
):
    """
Constrói envelopes empíricos dos surrogates por ativo e janela.
    """
    required_columns = [
        "asset",
        "window_id",
        "reference_date",
        "tp_h1",
        "surrogate_id",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in surrogates_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Colunas ausentes nos surrogates: {missing_columns}")

    grouped = surrogates_df.groupby(["asset", "window_id"], as_index=False)

    envelope_df = grouped.agg(
        reference_date=("reference_date", "first"),
        surrogate_count=("surrogate_id", "nunique"),
        surrogate_mean=("tp_h1", "mean"),
        surrogate_std=("tp_h1", "std"),
        surrogate_median=("tp_h1", "median"),
        surrogate_min=("tp_h1", "min"),
        surrogate_max=("tp_h1", "max"),
        envelope_lower=("tp_h1", lambda x: x.quantile(lower_quantile)),
        envelope_upper=("tp_h1", lambda x: x.quantile(upper_quantile)),
    )

    envelope_df["lower_quantile"] = lower_quantile
    envelope_df["upper_quantile"] = upper_quantile

    return envelope_df


def compare_real_with_envelope(real_df, envelope_df):
    """
    Compara o TP_H1 real com o envelope empírico dos surrogates.

    Adiciona flags de posição em relação ao envelope:
    - above_envelope
    - below_envelope
    - inside_envelope
    """
    required_real_columns = [
        "asset",
        "window_id",
        "reference_date",
        "tp_h1",
    ]

    missing_real_columns = [
        column for column in required_real_columns
        if column not in real_df.columns
    ]

    if missing_real_columns:
        raise ValueError(f"Colunas ausentes no TP_H1 real: {missing_real_columns}")

    real_selected = real_df[
        [
            "asset",
            "window_id",
            "reference_date",
            "window_start_date",
            "window_end_date",
            "tau",
            "embedding_dimension",
            "n_window_obs",
            "n_embedding_points",
            "homology_dimension",
            "tp_h1",
            "n_h1_features",
        ]
    ].copy()

    real_selected = real_selected.rename(
        columns={
            "tp_h1": "tp_h1_real",
            "n_h1_features": "n_h1_features_real",
        }
    )

    comparison_df = real_selected.merge(
        envelope_df,
        on=["asset", "window_id", "reference_date"],
        how="left",
        validate="one_to_one",
    )

    if comparison_df["envelope_lower"].isna().any():
        missing = comparison_df[comparison_df["envelope_lower"].isna()]
        raise ValueError(
            f"Há janelas reais sem envelope correspondente. Total: {len(missing)}"
        )

    comparison_df["above_envelope"] = (
        comparison_df["tp_h1_real"] > comparison_df["envelope_upper"]
    )

    comparison_df["below_envelope"] = (
        comparison_df["tp_h1_real"] < comparison_df["envelope_lower"]
    )

    comparison_df["inside_envelope"] = (
        (~comparison_df["above_envelope"])
        & (~comparison_df["below_envelope"])
    )

    comparison_df["distance_to_surrogate_mean"] = (
        comparison_df["tp_h1_real"] - comparison_df["surrogate_mean"]
    )

    comparison_df["z_score_vs_surrogates"] = (
        comparison_df["distance_to_surrogate_mean"]
        / comparison_df["surrogate_std"].replace(0, np.nan)
    )

    return comparison_df


def summarize_envelope_comparison(comparison_df):
    """
    Agrega por ativo os resultados da comparação entre TP_H1 real e envelope.
    """
    summary_df = (
        comparison_df
        .groupby("asset")
        .agg(
            n_windows=("window_id", "count"),
            n_above_envelope=("above_envelope", "sum"),
            n_below_envelope=("below_envelope", "sum"),
            n_inside_envelope=("inside_envelope", "sum"),
            mean_tp_h1_real=("tp_h1_real", "mean"),
            mean_surrogate_mean=("surrogate_mean", "mean"),
            mean_distance_to_surrogate_mean=(
                "distance_to_surrogate_mean",
                "mean",
            ),
            mean_z_score_vs_surrogates=("z_score_vs_surrogates", "mean"),
            max_z_score_vs_surrogates=("z_score_vs_surrogates", "max"),
            min_z_score_vs_surrogates=("z_score_vs_surrogates", "min"),
        )
        .reset_index()
    )

    summary_df["prop_above_envelope"] = (
        summary_df["n_above_envelope"] / summary_df["n_windows"]
    )

    summary_df["prop_below_envelope"] = (
        summary_df["n_below_envelope"] / summary_df["n_windows"]
    )

    summary_df["prop_inside_envelope"] = (
        summary_df["n_inside_envelope"] / summary_df["n_windows"]
    )

    return summary_df


def validate_envelope_outputs(envelope_df, comparison_df, summary_df):
    """
    Valida a consistência básica das saídas da etapa de envelopes.
    """
    if envelope_df.empty:
        raise ValueError("Envelope empírico vazio.")

    if comparison_df.empty:
        raise ValueError("Tabela real vs envelope vazia.")

    if summary_df.empty:
        raise ValueError("Resumo do envelope vazio.")

    if comparison_df["tp_h1_real"].isna().any():
        raise ValueError("Há NaN em tp_h1_real.")

    if comparison_df["envelope_lower"].isna().any():
        raise ValueError("Há NaN em envelope_lower.")

    if comparison_df["envelope_upper"].isna().any():
        raise ValueError("Há NaN em envelope_upper.")

    invalid_envelope = comparison_df[
        comparison_df["envelope_lower"] > comparison_df["envelope_upper"]
    ]

    if not invalid_envelope.empty:
        raise ValueError("Há envelopes com limite inferior maior que o superior.")

    flag_sum = (
        comparison_df["above_envelope"].astype(int)
        + comparison_df["below_envelope"].astype(int)
        + comparison_df["inside_envelope"].astype(int)
    )

    if not (flag_sum == 1).all():
        raise ValueError("Flags de envelope inconsistentes.")


