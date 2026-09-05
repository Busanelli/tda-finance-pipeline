import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import get_path, log_message, setup_project
from src.io_utils import (
    load_dataframe,
    load_time_indexed_dataframe,
    save_dataframe,
)
from src.persistence_diagrams import generate_persistence_diagrams_for_all_assets


def sanitize_asset_name(asset):
    """
    Converte o nome do ativo para uso seguro em arquivos e pastas.
    """
    return asset.replace(".", "_").replace("/", "_")


def create_output_dirs(project_root, config):
    """
    Cria a estrutura de diretórios dos outputs finais.
    """
    outputs_relative = config.get("paths", {}).get("outputs", "outputs")
    outputs_root = project_root / outputs_relative

    output_paths = {
        "root": outputs_root,
        "csv": outputs_root / "csv",
        "tables": outputs_root / "tables",
        "figures": outputs_root / "figures",
        "figures_tp_h1_envelope": outputs_root / "figures" / "tp_h1_vs_envelope",
        "figures_tp_h1_volatility": outputs_root / "figures" / "tp_h1_vs_volatility",
        "figures_persistence": outputs_root / "figures" / "persistence",
    }

    for path in output_paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)

    return output_paths


def standardize_result_columns(df):
    """
    Padroniza colunas e tipos básicos das tabelas de resultados.
    """
    df = df.copy()

    if "tp_h1" in df.columns and "tp_h1_real" not in df.columns:
        df = df.rename(columns={"tp_h1": "tp_h1_real"})

    date_columns = [
        "reference_date",
        "window_start_date",
        "window_end_date",
    ]
    for column in date_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column])

    flag_columns = ["above_envelope", "below_envelope", "inside_envelope"]
    for column in flag_columns:
        if column in df.columns:
            df[column] = df[column].fillna(False).astype(bool)

    return df


def merge_missing_columns(base_df, extra_df, key_columns):
    """
    Adiciona ao DataFrame-base colunas ausentes presentes em outro DataFrame.
    """
    extra_columns = [col for col in extra_df.columns if col not in base_df.columns]

    if not extra_columns:
        return base_df.copy()

    return base_df.merge(
        extra_df[key_columns + extra_columns],
        on=key_columns,
        how="left",
    )


def load_required_inputs(config):
    """
    Carrega os insumos necessários para a etapa final de outputs.
    """
    topology_path = get_path(config, "topology")
    processed_path = get_path(config, "processed")
    windows_path = get_path(config, "windows")
    tables_intermediate_path = get_path(config, "tables_intermediate")

    real_topology_path = topology_path / "tp_h1_real.parquet"
    comparison_path = topology_path / "tp_h1_real_vs_envelope.parquet"
    log_returns_path = processed_path / "log_returns.parquet"
    windows_input_path = windows_path / "windows_normalized.parquet"
    windows_metadata_path = tables_intermediate_path / "windows_metadata.parquet"
    parameters_input_path = tables_intermediate_path / "embedding_parameters.parquet"

    log_message(f"Carregando topologia real de: {real_topology_path}")
    real_df = load_dataframe(real_topology_path)
    real_df = standardize_result_columns(real_df)

    log_message(f"Carregando comparação real vs envelope de: {comparison_path}")
    comparison_df = load_dataframe(comparison_path)
    comparison_df = standardize_result_columns(comparison_df)

    log_message(f"Carregando retornos logarítmicos de: {log_returns_path}")
    log_returns = load_time_indexed_dataframe(log_returns_path)

    log_message(f"Carregando janelas normalizadas de: {windows_input_path}")
    windows_df = load_dataframe(windows_input_path)

    log_message(f"Carregando metadados das janelas de: {windows_metadata_path}")
    windows_metadata_df = load_dataframe(windows_metadata_path)

    log_message(f"Carregando parâmetros de embedding de: {parameters_input_path}")
    parameters_df = load_dataframe(parameters_input_path)

    return {
        "real_df": real_df,
        "comparison_df": comparison_df,
        "log_returns": log_returns,
        "windows_df": windows_df,
        "windows_metadata_df": windows_metadata_df,
        "parameters_df": parameters_df,
    }


def compute_window_volatility(log_returns, window_size, step_size, annual_trading_days=252):
    """
    Calcula a volatilidade por janela para todos os ativos.
    """
    records = []

    for asset in log_returns.columns:
        series = log_returns[asset].copy()

        if series.isna().any():
            raise ValueError(f"A série de retornos do ativo {asset} contém valores nulos.")

        for window_id, start in enumerate(range(0, len(series) - window_size + 1, step_size)):
            end = start + window_size
            window = series.iloc[start:end]

            volatility_window = float(window.std(ddof=1))
            volatility_annualized = float(volatility_window * np.sqrt(annual_trading_days))

            records.append(
                {
                    "asset": asset,
                    "window_id": window_id,
                    "reference_date": window.index[-1],
                    "volatility_window": volatility_window,
                    "volatility_annualized": volatility_annualized,
                }
            )

    volatility_df = pd.DataFrame(records)
    volatility_df["reference_date"] = pd.to_datetime(volatility_df["reference_date"])

    return volatility_df


def build_final_results_by_window(comparison_df, real_df, volatility_df):
    """
    Organiza a tabela final por janela.
    """
    key_columns = ["asset", "window_id", "reference_date"]

    final_df = merge_missing_columns(
        base_df=comparison_df,
        extra_df=real_df,
        key_columns=key_columns,
    )

    final_df = final_df.merge(
        volatility_df,
        on=key_columns,
        how="left",
        validate="one_to_one",
    )

    if (
        {"tp_h1_real", "surrogate_mean", "surrogate_std"}.issubset(final_df.columns)
        and "z_score_vs_surrogates" not in final_df.columns
    ):
        surrogate_std = final_df["surrogate_std"].replace(0, np.nan)
        final_df["z_score_vs_surrogates"] = (
            final_df["tp_h1_real"] - final_df["surrogate_mean"]
        ) / surrogate_std

    if "inside_envelope" not in final_df.columns:
        if {"above_envelope", "below_envelope"}.issubset(final_df.columns):
            final_df["inside_envelope"] = ~(
                final_df["above_envelope"] | final_df["below_envelope"]
            )

    final_df = final_df.sort_values(["asset", "reference_date"]).reset_index(drop=True)

    return final_df


def build_final_summary_by_asset(final_df):
    """
    Gera uma tabela-resumo final por ativo.
    """
    records = []

    for asset, group in final_df.groupby("asset"):
        group = group.sort_values("reference_date").copy()

        n_windows = int(len(group))
        n_above = int(group["above_envelope"].sum()) if "above_envelope" in group.columns else 0
        n_below = int(group["below_envelope"].sum()) if "below_envelope" in group.columns else 0
        n_inside = int(group["inside_envelope"].sum()) if "inside_envelope" in group.columns else 0

        valid_corr = group[["tp_h1_real", "volatility_window"]].dropna()

        if len(valid_corr) >= 2:
            rho, p_value = spearmanr(
                valid_corr["tp_h1_real"],
                valid_corr["volatility_window"],
            )
        else:
            rho, p_value = np.nan, np.nan

        records.append(
            {
                "asset": asset,
                "n_windows": n_windows,
                "mean_tp_h1": group["tp_h1_real"].mean(),
                "median_tp_h1": group["tp_h1_real"].median(),
                "std_tp_h1": group["tp_h1_real"].std(),
                "mean_volatility_window": group["volatility_window"].mean(),
                "mean_volatility_annualized": group["volatility_annualized"].mean(),
                "n_above_envelope": n_above,
                "n_below_envelope": n_below,
                "n_inside_envelope": n_inside,
                "prop_above_envelope": n_above / n_windows if n_windows > 0 else np.nan,
                "prop_below_envelope": n_below / n_windows if n_windows > 0 else np.nan,
                "prop_inside_envelope": n_inside / n_windows if n_windows > 0 else np.nan,
                "spearman_rho_tp_h1_volatility": rho,
                "p_value_tp_h1_volatility": p_value,
            }
        )

    summary_df = pd.DataFrame(records).sort_values("asset").reset_index(drop=True)

    return summary_df


def build_windows_above_envelope(final_df):
    """
    Seleciona as janelas em que TP_H1 real ficou acima do envelope.
    """
    if "above_envelope" not in final_df.columns:
        raise ValueError("A coluna 'above_envelope' não está disponível na tabela final.")

    return (
        final_df[final_df["above_envelope"]]
        .sort_values(["asset", "reference_date"])
        .reset_index(drop=True)
    )


def build_tp_h1_wide_table(final_df):
    """
    Constrói uma tabela larga das séries de TP_H1 real por ativo.
    """
    wide_df = (
        final_df.pivot_table(
            index="reference_date",
            columns="asset",
            values="tp_h1_real",
        )
        .sort_index()
    )

    return wide_df


def build_spearman_matrix_and_pairs(final_df):
    """
    Calcula a matriz de Spearman entre ativos e a tabela longa de pares.
    """
    wide_df = build_tp_h1_wide_table(final_df)

    matrix_df = wide_df.corr(method="spearman")

    pair_records = []
    assets = list(wide_df.columns)

    for asset_i, asset_j in combinations(assets, 2):
        aligned = wide_df[[asset_i, asset_j]].dropna()

        if len(aligned) >= 2:
            rho, p_value = spearmanr(aligned[asset_i], aligned[asset_j])
        else:
            rho, p_value = np.nan, np.nan

        pair_records.append(
            {
                "asset_i": asset_i,
                "asset_j": asset_j,
                "spearman_rho": rho,
                "p_value": p_value,
                "n_windows": len(aligned),
            }
        )

    pairs_df = pd.DataFrame(pair_records).sort_values(
        ["asset_i", "asset_j"]
    ).reset_index(drop=True)

    return matrix_df, pairs_df


def build_tp_h1_volatility_summary(final_df):
    """
    Gera um resumo da relação entre TP_H1 e volatilidade por ativo.
    """
    records = []

    for asset, group in final_df.groupby("asset"):
        group = group.sort_values("reference_date").copy()
        valid = group[["tp_h1_real", "volatility_window"]].dropna()

        if len(valid) >= 2:
            rho, p_value = spearmanr(
                valid["tp_h1_real"],
                valid["volatility_window"],
            )
        else:
            rho, p_value = np.nan, np.nan

        records.append(
            {
                "asset": asset,
                "n_windows": len(group),
                "mean_tp_h1": group["tp_h1_real"].mean(),
                "median_tp_h1": group["tp_h1_real"].median(),
                "mean_volatility_window": group["volatility_window"].mean(),
                "mean_volatility_annualized": group["volatility_annualized"].mean(),
                "spearman_rho_tp_h1_volatility": rho,
                "p_value_tp_h1_volatility": p_value,
            }
        )

    summary_df = pd.DataFrame(records).sort_values("asset").reset_index(drop=True)

    return summary_df


def save_figure(fig, output_path_base):
    """
    Salva uma figura em PNG e PDF.
    """
    output_path_base = Path(output_path_base)
    output_path_base.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output_path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_prop_above_envelope(summary_df, output_dir):
    """
    Gera gráfico de barras com a proporção de janelas acima do envelope por ativo.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    ax.bar(
        summary_df["asset"],
        summary_df["prop_above_envelope"],
    )

    ax.set_title("Proporção de janelas acima do envelope por ativo")
    ax.set_xlabel("Ativo")
    ax.set_ylabel("Proporção acima do envelope")
    ax.set_ylim(0, 1)

    output_path_base = Path(output_dir) / "prop_above_envelope_by_asset"
    save_figure(fig, output_path_base)


def plot_spearman_heatmap(matrix_df, output_dir):
    """
    Gera heatmap da matriz de correlação de Spearman entre ativos.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        matrix_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        ax=ax,
    )

    ax.set_title("Matriz de correlação de Spearman entre séries TP_H1")

    output_path_base = Path(output_dir) / "spearman_tp_h1_heatmap"
    save_figure(fig, output_path_base)


def plot_tp_h1_vs_envelope_for_asset(final_df, asset, output_dir):
    """
    Gera figura do TP_H1 real contra o envelope empírico para um ativo.
    """
    asset_df = (
        final_df[final_df["asset"] == asset]
        .sort_values("reference_date")
        .copy()
    )

    if asset_df.empty:
        raise ValueError(f"Nenhum dado encontrado para o ativo: {asset}")

    fig, ax = plt.subplots(figsize=(9, 4.5))

    x = asset_df["reference_date"]
    y = asset_df["tp_h1_real"]

    ax.plot(x, y, linewidth=1.3, label="TP_H1 real")

    if {"envelope_lower", "envelope_upper"}.issubset(asset_df.columns):
        ax.fill_between(
            x,
            asset_df["envelope_lower"],
            asset_df["envelope_upper"],
            alpha=0.25,
            label="Envelope empírico",
        )

    if "surrogate_median" in asset_df.columns:
        ax.plot(
            x,
            asset_df["surrogate_median"],
            linewidth=1.0,
            linestyle="--",
            label="Mediana surrogate",
        )

    if "above_envelope" in asset_df.columns:
        above_df = asset_df[asset_df["above_envelope"]]
        if not above_df.empty:
            ax.scatter(
                above_df["reference_date"],
                above_df["tp_h1_real"],
                s=18,
                label="Acima do envelope",
            )

    ax.set_title(f"TP_H1 real vs envelope — {asset}")
    ax.set_xlabel("Data de referência")
    ax.set_ylabel("TP_H1")
    ax.legend(frameon=False)

    safe_asset = sanitize_asset_name(asset)
    output_path_base = Path(output_dir) / f"tp_h1_vs_envelope_{safe_asset}"
    save_figure(fig, output_path_base)


def plot_tp_h1_vs_volatility_for_asset(final_df, asset, output_dir):
    """
    Gera figura do TP_H1 real e da volatilidade por janela para um ativo.
    """
    asset_df = (
        final_df[final_df["asset"] == asset]
        .sort_values("reference_date")
        .copy()
    )

    if asset_df.empty:
        raise ValueError(f"Nenhum dado encontrado para o ativo: {asset}")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    axes[0].plot(
        asset_df["reference_date"],
        asset_df["tp_h1_real"],
        linewidth=1.2,
    )
    axes[0].set_title(f"TP_H1 real — {asset}")
    axes[0].set_ylabel("TP_H1")

    axes[1].plot(
        asset_df["reference_date"],
        asset_df["volatility_window"],
        linewidth=1.2,
    )
    axes[1].set_title(f"Volatilidade por janela — {asset}")
    axes[1].set_xlabel("Data de referência")
    axes[1].set_ylabel("Volatilidade")

    safe_asset = sanitize_asset_name(asset)
    output_path_base = Path(output_dir) / f"tp_h1_vs_volatility_{safe_asset}"
    save_figure(fig, output_path_base)


def generate_all_figures(final_df, summary_df, matrix_df, output_paths):
    """
    Gera todas as figuras finais principais.
    """
    log_message("Gerando figura: proporção acima do envelope")
    plot_prop_above_envelope(
        summary_df=summary_df,
        output_dir=output_paths["figures"],
    )

    log_message("Gerando figura: heatmap de Spearman")
    plot_spearman_heatmap(
        matrix_df=matrix_df,
        output_dir=output_paths["figures"],
    )

    assets = sorted(final_df["asset"].unique())

    for asset in assets:
        log_message(f"Gerando figuras por ativo: {asset}")

        plot_tp_h1_vs_envelope_for_asset(
            final_df=final_df,
            asset=asset,
            output_dir=output_paths["figures_tp_h1_envelope"],
        )

        plot_tp_h1_vs_volatility_for_asset(
            final_df=final_df,
            asset=asset,
            output_dir=output_paths["figures_tp_h1_volatility"],
        )


def main():
    log_message("Iniciando etapa 08: geração dos outputs finais")

    config = setup_project()
    output_paths = create_output_dirs(PROJECT_ROOT, config)

    inputs = load_required_inputs(config)

    real_df = inputs["real_df"]
    comparison_df = inputs["comparison_df"]
    log_returns = inputs["log_returns"]
    windows_df = inputs["windows_df"]
    parameters_df = inputs["parameters_df"]
    windows_metadata_df = inputs["windows_metadata_df"]

    window_size = config["windowing"]["window_size"]
    step_size = config["windowing"]["step_size"]
    annual_trading_days = config.get("data", {}).get("annual_trading_days", 252)
    homology_dimension = config["topology"]["homology_dimension"]
    n_jobs = config.get("execution", {}).get("n_jobs", -1)

    log_message("Calculando volatilidade por janela")
    volatility_df = compute_window_volatility(
        log_returns=log_returns,
        window_size=window_size,
        step_size=step_size,
        annual_trading_days=annual_trading_days,
    )

    log_message("Montando tabela final por janela")
    final_results_df = build_final_results_by_window(
        comparison_df=comparison_df,
        real_df=real_df,
        volatility_df=volatility_df,
    )

    log_message("Montando resumo final por ativo")
    final_summary_df = build_final_summary_by_asset(final_results_df)

    log_message("Selecionando janelas acima do envelope")
    windows_above_df = build_windows_above_envelope(final_results_df)

    log_message("Calculando correlações de Spearman entre ativos")
    spearman_matrix_df, spearman_pairs_df = build_spearman_matrix_and_pairs(
        final_results_df
    )

    log_message("Construindo resumo TP_H1 vs volatilidade")
    tp_h1_volatility_summary_df = build_tp_h1_volatility_summary(final_results_df)

    log_message("Salvando outputs tabulares")
    save_dataframe(
        final_results_df,
        output_paths["csv"] / "final_results_by_window.csv",
    )
    save_dataframe(
        final_results_df,
        output_paths["tables"] / "final_results_by_window.parquet",
    )

    save_dataframe(
        final_summary_df,
        output_paths["csv"] / "final_summary_by_asset.csv",
    )
    save_dataframe(
        final_summary_df,
        output_paths["tables"] / "final_summary_by_asset.parquet",
    )

    save_dataframe(
        windows_above_df,
        output_paths["csv"] / "windows_above_envelope.csv",
    )
    save_dataframe(
        windows_above_df,
        output_paths["tables"] / "windows_above_envelope.parquet",
    )

    save_dataframe(
        spearman_pairs_df,
        output_paths["csv"] / "spearman_tp_h1_pairs.csv",
    )
    save_dataframe(
        spearman_pairs_df,
        output_paths["tables"] / "spearman_tp_h1_pairs.parquet",
    )

    save_dataframe(
        spearman_matrix_df,
        output_paths["csv"] / "spearman_tp_h1_matrix.csv",
        include_index=True,
    )
    save_dataframe(
        spearman_matrix_df,
        output_paths["tables"] / "spearman_tp_h1_matrix.parquet",
        include_index=True,
    )

    save_dataframe(
        tp_h1_volatility_summary_df,
        output_paths["csv"] / "tp_h1_volatility_summary.csv",
    )
    save_dataframe(
        tp_h1_volatility_summary_df,
        output_paths["tables"] / "tp_h1_volatility_summary.parquet",
    )

    log_message("Gerando figuras finais")
    generate_all_figures(
        final_df=final_results_df,
        summary_df=final_summary_df,
        matrix_df=spearman_matrix_df,
        output_paths=output_paths,
    )

    log_message("Gerando diagramas de persistência representativos por ativo")
    
    diagram_records_df = generate_persistence_diagrams_for_all_assets(
        final_results_df=final_results_df,
        windows_df=windows_df,
        windows_metadata_df=windows_metadata_df,
        parameters_df=parameters_df,
        output_dir=output_paths["figures_persistence"],
        homology_dimension=homology_dimension,
        n_jobs=n_jobs,
    )

    if not diagram_records_df.empty:
        diagram_records_df = diagram_records_df.copy()
        if "output_path" in diagram_records_df.columns:
            diagram_records_df["output_path"] = diagram_records_df["output_path"].astype(str)

        save_dataframe(
            diagram_records_df,
            output_paths["csv"] / "persistence_diagram_records.csv",
        )
        save_dataframe(
            diagram_records_df,
            output_paths["tables"] / "persistence_diagram_records.parquet",
        )

    log_message("Etapa 08 concluída com sucesso")


if __name__ == "__main__":
    main()