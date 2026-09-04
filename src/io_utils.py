from pathlib import Path

import pandas as pd


def save_dataframe(df, output_path, include_index=False):
    """
    Salva um DataFrame em .csv ou .parquet.

    O índice é salvo apenas quando include_index=True.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".csv":
        df.to_csv(output_path, index=include_index)
    elif output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=include_index)
    else:
        raise ValueError(
            "Formato não suportado. Use .csv ou .parquet."
        )

def load_dataframe(input_path):
    """
    Carrega um DataFrame em .csv ou .parquet.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")

    if input_path.suffix == ".csv":
        return pd.read_csv(input_path)

    if input_path.suffix == ".parquet":
        return pd.read_parquet(input_path)

    raise ValueError("Formato não suportado. Use .csv ou .parquet.")


def load_time_indexed_dataframe(input_path, index_name="Date"):
    """
    Carrega um DataFrame preservando um índice temporal.
    """
    df = load_dataframe(input_path)

    if index_name in df.columns:
        df = df.set_index(index_name)

    df.index = pd.to_datetime(df.index)
    df.index.name = index_name

    return df