from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.topology import (
    compute_persistence_diagrams,
    get_asset_parameters,
    takens_embedding,
)


def sanitize_asset_name(asset):
    """
    Converte o nome do ativo para uso seguro em arquivos e pastas.
    """
    return asset.replace(".", "_").replace("/", "_")


def select_max_tp_h1_window(final_results_df, asset):
    """
    Seleciona a janela de maior TP_H1 real para um ativo.
    """
    asset_df = final_results_df[final_results_df["asset"] == asset].copy()

    if asset_df.empty:
        raise ValueError(f"Ativo não encontrado nos resultados finais: {asset}")

    if "tp_h1_real" in asset_df.columns:
        tp_column = "tp_h1_real"
    elif "tp_h1" in asset_df.columns:
        tp_column = "tp_h1"
    else:
        raise KeyError(
            "Coluna de TP_H1 não encontrada. Esperado: 'tp_h1_real' ou 'tp_h1'."
        )

    return asset_df.loc[asset_df[tp_column].idxmax()]


def get_window_values(windows_df, asset, window_id):
    """
    Recupera os valores normalizados de uma janela.
    """
    row = windows_df[
        (windows_df["asset"] == asset)
        & (windows_df["window_id"] == window_id)
    ]

    if row.empty:
        raise ValueError(
            f"Janela não encontrada para asset={asset}, window_id={window_id}."
        )

    if len(row) > 1:
        raise ValueError(
            f"Mais de uma janela encontrada para asset={asset}, window_id={window_id}."
        )

    return np.asarray(row.iloc[0]["window_values_normalized"], dtype=float)


def plot_persistence_diagram(diagram, output_path, title=None):
    """
    Gera e salva um diagrama de persistência com escala de persistência.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    diagram = np.asarray(diagram, dtype=float)

    if diagram.size == 0:
        raise ValueError("Diagrama de persistência vazio.")

    births = diagram[:, 0]
    deaths = diagram[:, 1]
    dimensions = diagram[:, 2].astype(int)

    finite_mask = np.isfinite(births) & np.isfinite(deaths)
    positive_mask = deaths > births
    valid_mask = finite_mask & positive_mask

    births = births[valid_mask]
    deaths = deaths[valid_mask]
    dimensions = dimensions[valid_mask]

    if len(births) == 0:
        raise ValueError("Diagrama sem pontos válidos para plotagem.")

    persistences = deaths - births

    min_value = min(births.min(), deaths.min())
    max_value = max(births.max(), deaths.max())

    margin = 0.05 * (max_value - min_value)

    if margin == 0:
        margin = 0.01

    fig, ax = plt.subplots(figsize=(6, 5.5))

    scatter = ax.scatter(
        births,
        deaths,
        c=persistences,
        s=35,
        alpha=0.85,
    )

    ax.plot(
        [min_value - margin, max_value + margin],
        [min_value - margin, max_value + margin],
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label="Diagonal birth = death",
    )

    for dimension in sorted(set(dimensions)):
        count = int((dimensions == dimension).sum())
        ax.scatter(
            [],
            [],
            label=f"H{dimension} ({count})",
        )

    ax.set(
        title=title or "Diagrama de persistência",
        xlabel="Birth",
        ylabel="Death",
        xlim=(min_value - margin, max_value + margin),
        ylim=(min_value - margin, max_value + margin),
    )

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Persistência (death - birth)")

    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")

    plt.close(fig)

def get_reference_date(windows_metadata_df, asset, window_id):
    """
    Recupera a data de referência de uma janela real.
    """
    row = windows_metadata_df[
        (windows_metadata_df["asset"] == asset)
        & (windows_metadata_df["window_id"] == window_id)
    ]

    if row.empty:
        raise ValueError(
            f"Data de referência não encontrada para asset={asset}, window_id={window_id}."
        )

    if len(row) > 1:
        raise ValueError(
            f"Mais de uma data encontrada para asset={asset}, window_id={window_id}."
        )

    return pd.to_datetime(row.iloc[0]["reference_date"])    

def plot_h1_barcode(diagram, output_path, title=None, max_bars=80):
    """
    Gera e salva um barcode de persistência apenas para H1.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    diagram = np.asarray(diagram, dtype=float)

    h1 = diagram[diagram[:, 2] == 1]

    finite_mask = np.isfinite(h1[:, 0]) & np.isfinite(h1[:, 1])
    positive_mask = h1[:, 1] > h1[:, 0]
    h1 = h1[finite_mask & positive_mask]

    if len(h1) == 0:
        raise ValueError("Nenhuma barra H1 válida encontrada.")

    h1 = pd.DataFrame(h1, columns=["birth", "death", "dimension"])
    h1["persistence"] = h1["death"] - h1["birth"]

    h1 = (
        h1
        .sort_values("persistence", ascending=False)
        .head(max_bars)
        .sort_values("birth")
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for idx, row in h1.iterrows():
        ax.hlines(
            y=idx,
            xmin=row["birth"],
            xmax=row["death"],
            linewidth=1.8,
        )

    ax.set_title(title or "Barcode de persistência em H1")
    ax.set_xlabel("Escala")
    ax.set_ylabel("Características H1")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")

    plt.close(fig)


def generate_persistence_diagram_for_asset(
    asset,
    final_results_df,
    windows_df,
    windows_metadata_df,
    parameters_df,
    output_dir,
    homology_dimension=1,
    n_jobs=-1,
):
    """
    Gera o diagrama de persistência e o barcode H1 da janela de maior TP_H1 real.
    """
    selected_window = select_max_tp_h1_window(
        final_results_df=final_results_df,
        asset=asset,
    )

    window_id = int(selected_window["window_id"])

    reference_date = get_reference_date(
        windows_metadata_df=windows_metadata_df,
        asset=asset,
        window_id=window_id,
    )

    values = get_window_values(
        windows_df=windows_df,
        asset=asset,
        window_id=window_id,
    )

    tau, dimension = get_asset_parameters(
        parameters_df=parameters_df,
        asset=asset,
    )

    embedded = takens_embedding(
        values=values,
        tau=tau,
        dimension=dimension,
    )

    diagrams = compute_persistence_diagrams(
        point_clouds=np.asarray([embedded], dtype=float),
        homology_dimension=homology_dimension,
        n_jobs=n_jobs,
    )

    diagram = diagrams[0]
    safe_asset = sanitize_asset_name(asset)

    diagram_output_path = Path(output_dir) / f"persistence_diagram_{safe_asset}.png"
    barcode_output_path = Path(output_dir) / f"persistence_barcode_h1_{safe_asset}.png"

    diagram_title = (
        f"Diagrama de persistência — {asset}\n"
        f"janela {window_id} | referência {reference_date.date()}"
    )

    barcode_title = (
        f"Barcode de persistência em H1 — {asset}\n"
        f"janela {window_id} | referência {reference_date.date()}"
    )

    plot_persistence_diagram(
        diagram=diagram,
        output_path=diagram_output_path,
        title=diagram_title,
    )

    plot_h1_barcode(
        diagram=diagram,
        output_path=barcode_output_path,
        title=barcode_title,
        max_bars=80,
    )

    return {
        "asset": asset,
        "window_id": window_id,
        "reference_date": reference_date,
        "tau": tau,
        "embedding_dimension": dimension,
        "diagram_output_path": str(diagram_output_path),
        "barcode_output_path": str(barcode_output_path),
    }


def generate_persistence_diagrams_for_all_assets(
    final_results_df,
    windows_df,
    windows_metadata_df,
    parameters_df,
    output_dir,
    homology_dimension=1,
    n_jobs=-1,
):
    """
    Gera diagramas de persistência para a janela de maior TP_H1 de cada ativo.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []

    assets = sorted(final_results_df["asset"].unique())

    for asset in assets:
        record = generate_persistence_diagram_for_asset(
            asset=asset,
            final_results_df=final_results_df,
            windows_df=windows_df,
            windows_metadata_df=windows_metadata_df,
            parameters_df=parameters_df,
            output_dir=output_dir,
            homology_dimension=homology_dimension,
            n_jobs=n_jobs,
        )

        records.append(record)

    return pd.DataFrame(records)

