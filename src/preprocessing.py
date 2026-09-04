from pathlib import Path

import numpy as np
import pandas as pd
from src.io_utils import load_dataframe


def compute_log_returns(prices):
    """
    Calcula retornos logarítmicos a partir dos preços.

    Fórmula:
        r_t = log(P_t / P_{t-1})
    """
    log_returns = np.log(prices / prices.shift(1))
    log_returns = log_returns.dropna(how="all")
    log_returns.index.name = "Date"

    return log_returns


def validate_log_returns(log_returns):
    """
    Valida a consistência básica da base de retornos logarítmicos.
    """
    if log_returns.empty:
        raise ValueError("A base de retornos logarítmicos está vazia.")

    total_missing = log_returns.isna().sum().sum()

    if total_missing > 0:
        missing_by_asset = log_returns.isna().sum()
        print("Valores nulos por ativo:")
        print(missing_by_asset)
        raise ValueError("A base de retornos contém valores nulos.")

    total_infinite = np.isinf(log_returns.to_numpy()).sum()

    if total_infinite > 0:
        raise ValueError("A base de retornos contém valores infinitos.")

