from pathlib import Path

import numpy as np
import pandas as pd

from src.io_utils import save_dataframe
from src.windowing import build_windows_for_asset
from src.topology import compute_topology_for_asset


def sanitize_asset_name(asset):
    """
    Converte o nome do ativo para uso seguro em arquivos e pastas.
    """
    return asset.replace(".", "_").replace("/", "_")


def shuffle_series_globally(series, seed):
    """
    Embaralha globalmente uma série temporal, preservando o índice original.

    A ordem dos valores é destruída, mas as datas permanecem disponíveis
    para manter window_id e reference_date comparáveis.
    """
    rng = np.random.default_rng(seed)

    values = series.to_numpy(dtype=float).copy()
    shuffled_values = rng.permutation(values)

    shuffled_series = pd.Series(
        data=shuffled_values,
        index=series.index,
        name=series.name,
    )

    return shuffled_series


def build_surrogate_windows(
    series,
    asset,
    surrogate_id,
    window_size,
    step_size,
    seed,
):
    """
    Gera uma versão embaralhada da série e aplica o mesmo janelamento da série real.

    Retorna um DataFrame com janelas e metadados, pronto para o cálculo topológico.
    """
    shuffled_series = shuffle_series_globally(
        series=series,
        seed=seed,
    )

    windows_df, metadata_df = build_windows_for_asset(
        series=shuffled_series,
        asset=asset,
        window_size=window_size,
        step_size=step_size,
    )

    surrogate_windows_df = windows_df.merge(
        metadata_df,
        on=["asset", "window_id"],
        how="left",
        validate="one_to_one",
    )

    surrogate_windows_df.insert(1, "surrogate_id", surrogate_id)

    return surrogate_windows_df


def compute_surrogate_topology(
    series,
    asset,
    surrogate_id,
    parameters_df,
    window_size,
    step_size,
    homology_dimension,
    seed,
    n_jobs=-1,
):
    """
    Calcula TP_H1 para um surrogate de um único ativo.

    Fluxo:
    1. embaralha globalmente a série de retornos;
    2. aplica o mesmo janelamento da série real;
    3. normaliza cada janela;
    4. usa tau e m estimados para o ativo real;
    5. calcula TP_H1 por janela.
    """
    surrogate_windows_df = build_surrogate_windows(
        series=series,
        asset=asset,
        surrogate_id=surrogate_id,
        window_size=window_size,
        step_size=step_size,
        seed=seed,
    )

    result_df = compute_topology_for_asset(
        asset_windows_df=surrogate_windows_df,
        parameters_df=parameters_df,
        asset=asset,
        homology_dimension=homology_dimension,
        n_jobs=n_jobs,
    )

    result_df.insert(1, "surrogate_id", surrogate_id)
    result_df.insert(2, "surrogate_type", "global_shuffle")
    result_df["seed"] = seed

    return result_df


def validate_surrogate_result(result_df):
    """
    Valida a consistência básica do TP_H1 calculado para um surrogate.
    """
    if result_df.empty:
        raise ValueError("Resultado do surrogate está vazio.")

    required_columns = [
        "asset",
        "surrogate_id",
        "surrogate_type",
        "window_id",
        "reference_date",
        "tau",
        "embedding_dimension",
        "tp_h1",
        "n_h1_features",
        "seed",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in result_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Colunas ausentes no surrogate: {missing_columns}")

    if result_df["tp_h1"].isna().any():
        raise ValueError("Há NaN em tp_h1 no surrogate.")

    if np.isinf(result_df["tp_h1"].to_numpy()).any():
        raise ValueError("Há infinito em tp_h1 no surrogate.")

    if (result_df["tp_h1"] < 0).any():
        raise ValueError("Há tp_h1 negativo no surrogate.")


def get_surrogate_output_path(partial_dir, asset, surrogate_id):
    """
Retorna o caminho do arquivo Parquet de um surrogate individual, organizado por ativo.
    """
    safe_asset = sanitize_asset_name(asset)

    return (
        Path(partial_dir)
        / safe_asset
        / f"surrogate_{surrogate_id:03d}.parquet"
    )


def compute_surrogates_for_asset(
    log_returns,
    parameters_df,
    asset,
    n_surrogates,
    window_size,
    step_size,
    homology_dimension,
    base_seed,
    partial_dir,
    n_jobs=-1,
    logger=None,
):
    """
    Calcula os surrogates de um ativo com salvamento incremental.

    Cada surrogate é salvo separadamente; arquivos já existentes são pulados.
    """
    series = log_returns[asset].dropna()
    partial_paths = []
    error_records = []

    for surrogate_id in range(n_surrogates):
        output_path = get_surrogate_output_path(
            partial_dir=partial_dir,
            asset=asset,
            surrogate_id=surrogate_id,
        )

        partial_paths.append(output_path)

        if output_path.exists():
            if logger is not None:
                logger(
                    f"{asset} | surrogate {surrogate_id:03d} já existe. Pulando."
                )
            continue

        seed = int(base_seed + surrogate_id)

        try:
            if logger is not None:
                logger(
                    f"{asset} | iniciando surrogate {surrogate_id:03d} | seed={seed}"
                )

            result_df = compute_surrogate_topology(
                series=series,
                asset=asset,
                surrogate_id=surrogate_id,
                parameters_df=parameters_df,
                window_size=window_size,
                step_size=step_size,
                homology_dimension=homology_dimension,
                seed=seed,
                n_jobs=n_jobs,
            )

            validate_surrogate_result(result_df)

            save_dataframe(result_df, output_path)

            if logger is not None:
                logger(
                    f"{asset} | surrogate {surrogate_id:03d} salvo em {output_path}"
                )

        except Exception as error:
            error_record = {
                "asset": asset,
                "surrogate_id": surrogate_id,
                "seed": seed,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }

            error_records.append(error_record)

            if logger is not None:
                logger(
                    f"{asset} | erro no surrogate {surrogate_id:03d}: {error}"
                )

    return partial_paths, pd.DataFrame(error_records)


def consolidate_surrogates(partial_dir):
    """
    Consolida os arquivos Parquet dos surrogates individuais em um único DataFrame.
    """
    partial_dir = Path(partial_dir)

    if not partial_dir.exists():
        raise FileNotFoundError(
            f"Diretório de surrogates não encontrado: {partial_dir}"
        )

    files = sorted(partial_dir.glob("*/*.parquet"))

    if len(files) == 0:
        raise FileNotFoundError(
            f"Nenhum arquivo de surrogate encontrado em: {partial_dir}"
        )

    dfs = [pd.read_parquet(file) for file in files]

    return pd.concat(dfs, ignore_index=True)