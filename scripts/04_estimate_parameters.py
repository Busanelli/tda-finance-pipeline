import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import get_path, log_message, setup_project
from src.io_utils import load_time_indexed_dataframe, save_dataframe
from src.parameters import estimate_parameters_for_all_assets


def main():
    log_message("Iniciando etapa 04: estimação de tau e dimensão de embedding")

    config = setup_project()

    processed_path = get_path(config, "processed")
    tables_intermediate_path = get_path(config, "tables_intermediate")

    input_path = processed_path / "log_returns.parquet"

    parameters_csv_path = tables_intermediate_path / "embedding_parameters.csv"
    parameters_parquet_path = tables_intermediate_path / "embedding_parameters.parquet"

    ami_csv_path = tables_intermediate_path / "ami_values.csv"
    ami_parquet_path = tables_intermediate_path / "ami_values.parquet"

    acf_csv_path = tables_intermediate_path / "acf_values.csv"
    acf_parquet_path = tables_intermediate_path / "acf_values.parquet"

    fnn_csv_path = tables_intermediate_path / "fnn_values.csv"
    fnn_parquet_path = tables_intermediate_path / "fnn_values.parquet"

    log_message(f"Carregando retornos logarítmicos de: {input_path}")
    log_returns = load_time_indexed_dataframe(input_path)

    log_message(f"Base de retornos carregada com shape: {log_returns.shape}")

    log_message("Estimando parâmetros por ativo")
    (
        parameters_df,
        ami_values_df,
        acf_values_df,
        fnn_values_df,
    ) = estimate_parameters_for_all_assets(
        log_returns=log_returns,
        config=config,
    )

    log_message("Parâmetros estimados:")
    print(parameters_df)

    log_message(f"Salvando parâmetros em CSV: {parameters_csv_path}")
    save_dataframe(parameters_df, parameters_csv_path)

    log_message(f"Salvando parâmetros em Parquet: {parameters_parquet_path}")
    save_dataframe(parameters_df, parameters_parquet_path)

    log_message(f"Salvando curva AMI em CSV: {ami_csv_path}")
    save_dataframe(ami_values_df, ami_csv_path)

    log_message(f"Salvando curva AMI em Parquet: {ami_parquet_path}")
    save_dataframe(ami_values_df, ami_parquet_path)

    log_message(f"Salvando curva ACF em CSV: {acf_csv_path}")
    save_dataframe(acf_values_df, acf_csv_path)

    log_message(f"Salvando curva ACF em Parquet: {acf_parquet_path}")
    save_dataframe(acf_values_df, acf_parquet_path)

    log_message(f"Salvando curva FNN em CSV: {fnn_csv_path}")
    save_dataframe(fnn_values_df, fnn_csv_path)

    log_message(f"Salvando curva FNN em Parquet: {fnn_parquet_path}")
    save_dataframe(fnn_values_df, fnn_parquet_path)

    log_message("Etapa 04 concluída com sucesso")


if __name__ == "__main__":
    main()