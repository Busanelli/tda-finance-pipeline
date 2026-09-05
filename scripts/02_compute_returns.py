import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import get_path, log_message, setup_project
from src.io_utils import load_time_indexed_dataframe, save_dataframe
from src.preprocessing import (
    compute_log_returns,
    validate_log_returns,
)


def main():
    log_message("Iniciando etapa 02: cálculo dos retornos logarítmicos")

    config = setup_project()

    raw_path = get_path(config, "raw")
    processed_path = get_path(config, "processed")

    input_path = raw_path / "prices.parquet"

    parquet_output_path = processed_path / "log_returns.parquet"
    csv_output_path = processed_path / "log_returns.csv"

    log_message(f"Carregando preços de: {input_path}")
    prices = load_time_indexed_dataframe(input_path)

    log_message(f"Base de preços carregada com shape: {prices.shape}")

    log_message("Calculando retornos logarítmicos")
    log_returns = compute_log_returns(prices)

    log_message(f"Base de retornos calculada com shape: {log_returns.shape}")

    log_message("Validando retornos logarítmicos")
    validate_log_returns(log_returns)

    log_message(f"Salvando retornos em Parquet: {parquet_output_path}")
    save_dataframe(log_returns, parquet_output_path, include_index=True)

    log_message(f"Salvando retornos em CSV: {csv_output_path}")
    save_dataframe(log_returns, csv_output_path)

    log_message("Etapa 02 concluída com sucesso")


if __name__ == "__main__":
    main()
    