import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import get_path, log_message, setup_project
from src.io_utils import load_dataframe, save_dataframe
from src.envelopes import (
    build_empirical_envelope,
    compare_real_with_envelope,
    summarize_envelope_comparison,
    validate_envelope_outputs,
)


def main():
    log_message("Iniciando etapa 07: construção do envelope empírico")

    config = setup_project()

    topology_path = get_path(config, "topology")
    surrogates_path = get_path(config, "surrogates")
    tables_final_path = get_path(config, "tables_final")

    real_input_path = topology_path / "tp_h1_real.parquet"
    surrogates_input_path = surrogates_path / "tp_h1_surrogates.parquet"

    envelope_parquet_path = topology_path / "tp_h1_envelope.parquet"
    envelope_csv_path = tables_final_path / "tp_h1_envelope.csv"

    comparison_parquet_path = topology_path / "tp_h1_real_vs_envelope.parquet"
    comparison_csv_path = tables_final_path / "tp_h1_real_vs_envelope.csv"

    summary_csv_path = tables_final_path / "envelope_summary.csv"
    summary_parquet_path = tables_final_path / "envelope_summary.parquet"

    lower_quantile = config.get("envelope", {}).get("lower_quantile", 0.025)
    upper_quantile = config.get("envelope", {}).get("upper_quantile", 0.975)

    log_message(f"Carregando TP_H1 real de: {real_input_path}")
    real_df = load_dataframe(real_input_path)

    log_message(f"Carregando surrogates de: {surrogates_input_path}")
    surrogates_df = load_dataframe(surrogates_input_path)

    log_message(f"TP_H1 real carregado com shape: {real_df.shape}")
    log_message(f"Surrogates carregados com shape: {surrogates_df.shape}")

    log_message(
        f"Construindo envelope empírico "
        f"com quantis {lower_quantile} e {upper_quantile}"
    )

    envelope_df = build_empirical_envelope(
        surrogates_df=surrogates_df,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )

    log_message(f"Envelope construído com shape: {envelope_df.shape}")

    log_message("Comparando TP_H1 real com envelope empírico")
    comparison_df = compare_real_with_envelope(
        real_df=real_df,
        envelope_df=envelope_df,
    )

    log_message(f"Tabela real vs envelope com shape: {comparison_df.shape}")

    log_message("Gerando resumo por ativo")
    summary_df = summarize_envelope_comparison(comparison_df)

    log_message("Resumo do envelope:")
    print(summary_df)

    log_message("Validando saídas da etapa 07")
    validate_envelope_outputs(
        envelope_df=envelope_df,
        comparison_df=comparison_df,
        summary_df=summary_df,
    )

    log_message(f"Salvando envelope em Parquet: {envelope_parquet_path}")
    save_dataframe(envelope_df, envelope_parquet_path)

    log_message(f"Salvando envelope em CSV: {envelope_csv_path}")
    save_dataframe(envelope_df, envelope_csv_path)

    log_message(f"Salvando comparação em Parquet: {comparison_parquet_path}")
    save_dataframe(comparison_df, comparison_parquet_path)

    log_message(f"Salvando comparação em CSV: {comparison_csv_path}")
    save_dataframe(comparison_df, comparison_csv_path)

    log_message(f"Salvando resumo em CSV: {summary_csv_path}")
    save_dataframe(summary_df, summary_csv_path)

    log_message(f"Salvando resumo em Parquet: {summary_parquet_path}")
    save_dataframe(summary_df, summary_parquet_path)

    log_message("Etapa 07 concluída com sucesso")


if __name__ == "__main__":
    main()