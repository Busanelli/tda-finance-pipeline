import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import get_path, log_message, setup_project
from src.io_utils import load_time_indexed_dataframe, save_dataframe
from src.windowing import (
    build_windows_for_all_assets,
    validate_windows,
)


def main():
    log_message("Iniciando etapa 03: janelamento e normalização")

    config = setup_project()

    processed_path = get_path(config, "processed")
    windows_path = get_path(config, "windows")
    tables_intermediate_path = get_path(config, "tables_intermediate")

    input_path = processed_path / "log_returns.parquet"

    windows_parquet_path = windows_path / "windows_normalized.parquet"
    metadata_csv_path = tables_intermediate_path / "windows_metadata.csv"
    metadata_parquet_path = tables_intermediate_path / "windows_metadata.parquet"

    window_size = config["windowing"]["window_size"]
    step_size = config["windowing"]["step_size"]

    log_message(f"Carregando retornos logarítmicos de: {input_path}")
    log_returns = load_time_indexed_dataframe(input_path)

    log_message(f"Base de retornos carregada com shape: {log_returns.shape}")

    log_message(
        f"Criando janelas com window_size={window_size} e step_size={step_size}"
    )

    windows_df, metadata_df = build_windows_for_all_assets(
        log_returns=log_returns,
        window_size=window_size,
        step_size=step_size,
    )

    log_message(f"Janelas criadas: {windows_df.shape[0]}")
    log_message(f"Metadados criados: {metadata_df.shape[0]}")

    log_message("Validando janelas")
    validate_windows(
        windows_df=windows_df,
        metadata_df=metadata_df,
        expected_window_size=window_size,
    )

    log_message(f"Salvando janelas normalizadas em: {windows_parquet_path}")
    save_dataframe(windows_df, windows_parquet_path)

    log_message(f"Salvando metadados em CSV: {metadata_csv_path}")
    save_dataframe(metadata_df, metadata_csv_path)

    log_message(f"Salvando metadados em Parquet: {metadata_parquet_path}")
    save_dataframe(metadata_df, metadata_parquet_path)

    log_message("Etapa 03 concluída com sucesso")


if __name__ == "__main__":
    main()