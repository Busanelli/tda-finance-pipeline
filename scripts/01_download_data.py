import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import get_path, log_message, setup_project
from src.data_loader import download_prices_for_assets
from src.io_utils import save_dataframe


def main():
    log_message("Iniciando etapa 01: download dos dados")

    config = setup_project()
    raw_path = get_path(config, "raw")

    parquet_path = raw_path / "prices.parquet"
    csv_path = raw_path / "prices.csv"

    log_message("Baixando preços ajustados dos ativos")
    prices = download_prices_for_assets(config)

    total_missing = prices.isna().sum().sum()

    if total_missing > 0:
        missing_by_asset = prices.isna().sum()
        log_message("Foram encontrados valores nulos nos preços")
        print(missing_by_asset)
        raise ValueError("A base de preços contém valores nulos.")

    log_message(f"Base baixada com shape: {prices.shape}")

    log_message(f"Salvando arquivo Parquet em: {parquet_path}")
    save_dataframe(prices, parquet_path)

    log_message(f"Salvando arquivo CSV em: {csv_path}")
    save_dataframe(prices, csv_path)

    log_message("Etapa 01 concluída com sucesso")


if __name__ == "__main__":
    main()