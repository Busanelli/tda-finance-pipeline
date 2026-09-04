from pathlib import Path

import numpy as np
import pandas as pd


def normalize_window(values):
    """
    Normaliza uma janela por z-score.

    Usa ddof=0 para calcular o desvio-padrão dentro da própria janela.
    """
    values = np.asarray(values, dtype=float)

    mean = values.mean()
    std = values.std(ddof=0)

    if std == 0:
        raise ValueError(
            "Janela com desvio-padrão zero. Não é possível normalizar."
        )

    return (values - mean) / std


def build_windows_for_asset(series, asset, window_size, step_size):
    """
    Cria janelas para um ativo e retorna as janelas normalizadas com seus metadados.
    """
    series = series.dropna()

    values = series.to_numpy(dtype=float)
    dates = series.index.to_numpy()

    n_obs = len(values)

    if n_obs < window_size:
        raise ValueError(
            f"Série do ativo {asset} tem menos observações que window_size."
        )

    windows_records = []
    metadata_records = []

    window_id = 0

    for start_position in range(0, n_obs - window_size + 1, step_size):
        end_position = start_position + window_size

        window_values = values[start_position:end_position]
        window_dates = dates[start_position:end_position]

        normalized_values = normalize_window(window_values)

        window_start_date = pd.Timestamp(window_dates[0])
        window_end_date = pd.Timestamp(window_dates[-1])
        reference_date = window_end_date

        windows_records.append(
            {
                "asset": asset,
                "window_id": window_id,
                "window_values": window_values.tolist(),
                "window_values_normalized": normalized_values.tolist(),
            }
        )

        metadata_records.append(
            {
                "asset": asset,
                "window_id": window_id,
                "start_position": start_position,
                "end_position": end_position - 1,
                "window_start_date": window_start_date,
                "window_end_date": window_end_date,
                "reference_date": reference_date,
                "n_obs": window_size,
                "window_size": window_size,
                "step_size": step_size,
            }
        )

        window_id += 1

    windows_df = pd.DataFrame(windows_records)
    metadata_df = pd.DataFrame(metadata_records)

    return windows_df, metadata_df


def build_windows_for_all_assets(log_returns, window_size, step_size):
    """
    Cria janelas para todos os ativos, retornando as janelas e seus metadados.
    """
    windows_list = []
    metadata_list = []

    for asset in log_returns.columns:
        windows_df, metadata_df = build_windows_for_asset(
            series=log_returns[asset],
            asset=asset,
            window_size=window_size,
            step_size=step_size,
        )

        windows_list.append(windows_df)
        metadata_list.append(metadata_df)

    all_windows = pd.concat(windows_list, ignore_index=True)
    all_metadata = pd.concat(metadata_list, ignore_index=True)

    return all_windows, all_metadata


def validate_windows(windows_df, metadata_df, expected_window_size):
    """
    Valida a consistência básica das janelas e dos metadados criados.

    """
    if windows_df.empty:
        raise ValueError("A tabela de janelas está vazia.")

    if metadata_df.empty:
        raise ValueError("A tabela de metadados está vazia.")

    if len(windows_df) != len(metadata_df):
        raise ValueError("Número de janelas diferente do número de metadados.")

    for column in ["window_values", "window_values_normalized"]:
        lengths = windows_df[column].apply(len)

        if not (lengths == expected_window_size).all():
            raise ValueError(
                f"Há janelas em {column} com tamanho diferente de {expected_window_size}."
            )

    for column in ["window_values", "window_values_normalized"]:
        has_nan = windows_df[column].apply(
            lambda values: np.isnan(np.asarray(values, dtype=float)).any()
        )

        if has_nan.any():
            raise ValueError(f"Há valores NaN em {column}.")

    required_metadata_columns = [
        "asset",
        "window_id",
        "start_position",
        "end_position",
        "window_start_date",
        "window_end_date",
        "reference_date",
        "n_obs",
        "window_size",
        "step_size",
    ]

    missing_columns = [
        column for column in required_metadata_columns
        if column not in metadata_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Colunas ausentes nos metadados: {missing_columns}")


