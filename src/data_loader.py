from pathlib import Path

import pandas as pd
import yfinance as yf


def download_asset_data(
    ticker,
    start_date,
    end_date,
    auto_adjust=False,
):
    """
    Baixa pelo yfinance os preços de um ativo no período informado.
    """
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
        progress=True,
    )

    if data.empty:
        raise ValueError(f"Nenhum dado foi baixado para o ativo: {ticker}")

    return data


def extract_price_series(data, ticker, price_column="Adj Close"):
    """
   Extrai a coluna de preços de um DataFrame do yfinance.

   Também trata retornos com colunas MultiIndex, como ocorre ao baixar múltiplos ativos.
    """
    if isinstance(data.columns, pd.MultiIndex):
        series = data[(price_column, ticker)].copy()
    else:
        series = data[price_column].copy()

    series.name = ticker

    return series


def download_prices_for_assets(config):
    """
    Baixa e consolida as séries de preços dos ativos definidos no config.
    """
    assets = config["data"]["assets"]
    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]
    price_column = config["data"]["price_column"]
    auto_adjust = config["data"]["auto_adjust"]

    price_series = []

    for ticker in assets:
        print(f"Baixando dados de {ticker}...")

        data = download_asset_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            auto_adjust=auto_adjust,
        )

        series = extract_price_series(
            data=data,
            ticker=ticker,
            price_column=price_column,
        )

        n_missing = series.isna().sum()
        print(f"{ticker}: {n_missing} valores nulos em {price_column}")

        price_series.append(series)

    prices = pd.concat(price_series, axis=1)
    prices.index.name = "Date"

    return prices


