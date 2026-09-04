import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import get_path, log_message, setup_project
from src.io_utils import load_dataframe, save_dataframe
from src.surrogates import (
    compute_surrogates_for_asset,
    consolidate_surrogates,
)


def main():
    log_message("Iniciando etapa 06: cálculo dos surrogates por embaralhamento global")

    config = setup_project()

    processed_path = get_path(config, "processed")
    surrogates_path = get_path(config, "surrogates")
    tables_intermediate_path = get_path(config, "tables_intermediate")

    log_returns_input_path = processed_path / "log_returns.parquet"
    parameters_input_path = tables_intermediate_path / "embedding_parameters.parquet"

    partial_dir = surrogates_path / "partial"

    output_parquet_path = surrogates_path / "tp_h1_surrogates.parquet"
    output_csv_path = tables_intermediate_path / "tp_h1_surrogates.csv"
    errors_csv_path = tables_intermediate_path / "surrogate_errors.csv"

    assets = config["data"]["assets"]

    window_size = config["windowing"]["window_size"]
    step_size = config["windowing"]["step_size"]

    homology_dimension = config["topology"]["homology_dimension"]

    n_surrogates = config["surrogates"]["n_surrogates"]
    random_seed = config["surrogates"]["random_seed"]

    n_jobs = config.get("execution", {}).get("n_jobs", -1)

    log_message(f"Carregando retornos logarítmicos de: {log_returns_input_path}")
    log_returns = load_dataframe(log_returns_input_path)

    log_message(f"Carregando parâmetros de embedding de: {parameters_input_path}")
    parameters_df = load_dataframe(parameters_input_path)

    log_message(f"Base de retornos carregada com shape: {log_returns.shape}")
    log_message(f"Parâmetros carregados com shape: {parameters_df.shape}")

    all_error_dfs = []

    for asset_index, asset in enumerate(assets):
        asset_seed = int(random_seed + asset_index * 100_000)

        log_message(
            f"Iniciando ativo {asset} | "
            f"n_surrogates={n_surrogates} | "
            f"base_seed={asset_seed}"
        )

        _, error_df = compute_surrogates_for_asset(
            log_returns=log_returns,
            parameters_df=parameters_df,
            asset=asset,
            n_surrogates=n_surrogates,
            window_size=window_size,
            step_size=step_size,
            homology_dimension=homology_dimension,
            base_seed=asset_seed,
            partial_dir=partial_dir,
            n_jobs=n_jobs,
            logger=log_message,
        )

        if not error_df.empty:
            all_error_dfs.append(error_df)

        log_message(f"Ativo {asset} concluído")

    if all_error_dfs:
        errors_df = pd.concat(all_error_dfs, ignore_index=True)
        log_message(f"Salvando relatório de erros em: {errors_csv_path}")
        save_dataframe(errors_df, errors_csv_path)
    else:
        log_message("Nenhum erro registrado nos surrogates")

    log_message("Consolidando arquivos parciais dos surrogates")
    surrogates_df = consolidate_surrogates(partial_dir)

    log_message(f"Surrogates consolidados com shape: {surrogates_df.shape}")

    log_message(f"Salvando surrogates consolidados em Parquet: {output_parquet_path}")
    save_dataframe(surrogates_df, output_parquet_path)

    log_message(f"Salvando surrogates consolidados em CSV: {output_csv_path}")
    save_dataframe(surrogates_df, output_csv_path)

    log_message("Etapa 06 concluída com sucesso")


if __name__ == "__main__":
    main()