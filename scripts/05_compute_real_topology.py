import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import get_path, log_message, setup_project
from src.io_utils import load_dataframe, save_dataframe
from src.topology import (
    compute_real_topology,
    validate_topology_results,
)


def main():
    log_message("Iniciando etapa 05: cálculo da topologia real")

    config = setup_project()

    windows_path = get_path(config, "windows")
    topology_path = get_path(config, "topology")
    tables_intermediate_path = get_path(config, "tables_intermediate")

    windows_input_path = windows_path / "windows_normalized.parquet"
    metadata_input_path = tables_intermediate_path / "windows_metadata.parquet"
    parameters_input_path = tables_intermediate_path / "embedding_parameters.parquet"

    output_parquet_path = topology_path / "tp_h1_real.parquet"
    output_csv_path = tables_intermediate_path / "tp_h1_real.csv"

    homology_dimension = config["topology"]["homology_dimension"]
    n_jobs = config.get("execution", {}).get("n_jobs", -1)

    log_message(f"Carregando janelas de: {windows_input_path}")
    windows_df = load_dataframe(windows_input_path)

    log_message(f"Carregando metadados de: {metadata_input_path}")
    metadata_df = load_dataframe(metadata_input_path)

    log_message(f"Carregando parâmetros de embedding de: {parameters_input_path}")
    parameters_df = load_dataframe(parameters_input_path)

    log_message(f"Janelas carregadas: {windows_df.shape}")
    log_message(f"Metadados carregados: {metadata_df.shape}")
    log_message(f"Parâmetros carregados: {parameters_df.shape}")

    log_message("Calculando TP_H1 real")
    results_df = compute_real_topology(
        windows_df=windows_df,
        metadata_df=metadata_df,
        parameters_df=parameters_df,
        homology_dimension=homology_dimension,
        n_jobs=n_jobs,
    )

    log_message(f"Resultados topológicos calculados: {results_df.shape}")

    log_message("Validando resultados topológicos")
    validate_topology_results(results_df)

    log_message(f"Salvando resultados em Parquet: {output_parquet_path}")
    save_dataframe(results_df, output_parquet_path)

    log_message(f"Salvando resultados em CSV: {output_csv_path}")
    save_dataframe(results_df, output_csv_path)

    log_message("Etapa 05 concluída com sucesso")


if __name__ == "__main__":
    main()