from pathlib import Path

import numpy as np
import pandas as pd

from tqdm import tqdm
from gtda.time_series import SingleTakensEmbedding
from gtda.homology import VietorisRipsPersistence


def takens_embedding(values, tau, dimension):
    """
    Cria o embedding de Takens para uma janela já definida.

    O janelamento temporal é feito antes desta etapa; aqui o giotto-tda
    recebe apenas os valores de uma janela individual.
    """
    values = np.asarray(values, dtype=float)

    if len(values) - (dimension - 1) * tau <= 1:
        raise ValueError(
            f"Janela curta demais para tau={tau} e dimension={dimension}."
        )

    embedder = SingleTakensEmbedding(
        parameters_type="fixed",
        time_delay=tau,
        dimension=dimension,
        stride=1,
    )

    return embedder.fit_transform(values)


def compute_tp_h1_from_diagram(diagram, homology_dimension=1):
    """
    Calcula a persistência total em H1 a partir de um diagrama do giotto-tda.

    Retorna a soma das persistências death - birth em H1 e o número de
    características H1 válidas.
    """
    diagram = np.asarray(diagram, dtype=float)

    if diagram.size == 0:
        return 0.0, 0

    h_mask = diagram[:, 2] == homology_dimension
    h_diagram = diagram[h_mask]

    if h_diagram.size == 0:
        return 0.0, 0

    births = h_diagram[:, 0]
    deaths = h_diagram[:, 1]

    finite_mask = np.isfinite(births) & np.isfinite(deaths)
    positive_mask = deaths > births

    valid_mask = finite_mask & positive_mask

    if not valid_mask.any():
        return 0.0, 0

    persistences = deaths[valid_mask] - births[valid_mask]

    tp_h1 = float(persistences.sum())
    n_h1_features = int(len(persistences))

    return tp_h1, n_h1_features


def prepare_windows_with_metadata(windows_df, metadata_df):
    """
    Junta as janelas normalizadas aos metadados correspondentes por ativo e window_id.

    """
    key_columns = ["asset", "window_id"]

    merged_df = windows_df.merge(
        metadata_df,
        on=key_columns,
        how="left",
        validate="one_to_one",
    )

    if merged_df["reference_date"].isna().any():
        raise ValueError("Há janelas sem reference_date após o merge.")

    return merged_df


def get_asset_parameters(parameters_df, asset):
    """
    Recupera tau e m finais de um ativo a partir da tabela de parâmetros.
    """
    row = parameters_df[parameters_df["asset"] == asset]

    if row.empty:
        raise ValueError(f"Parâmetros não encontrados para o ativo: {asset}")

    if len(row) > 1:
        raise ValueError(f"Mais de uma linha de parâmetros para o ativo: {asset}")

    tau = int(row.iloc[0]["tau_final"])
    dimension = int(row.iloc[0]["m_final"])

    return tau, dimension


def compute_topology_for_asset(
    asset_windows_df,
    parameters_df,
    asset,
    homology_dimension=1,
    n_jobs=-1,
):
    """
    Calcula TP_H1 para todas as janelas de um ativo.

    Fluxo:
    1. recupera tau e m estimados para o ativo;
    2. aplica o embedding de Takens em cada janela normalizada;
    3. calcula homologia persistente em H1 com Vietoris-Rips;
    4. soma as persistências H1 de cada janela.
    """
    tau, dimension = get_asset_parameters(parameters_df, asset)

    point_clouds = []
    n_embedding_points_list = []

    for _, row in asset_windows_df.iterrows():
        values = row["window_values_normalized"]

        embedded = takens_embedding(
            values=values,
            tau=tau,
            dimension=dimension,
        )

        point_clouds.append(embedded)
        n_embedding_points_list.append(embedded.shape[0])

    n_unique_embedding_sizes = len(set(n_embedding_points_list))

    if n_unique_embedding_sizes != 1:
        raise ValueError(
            f"As janelas do ativo {asset} geraram embeddings com tamanhos diferentes."
        )

    point_clouds = np.stack(point_clouds, axis=0)

    diagrams = compute_persistence_diagrams(
    point_clouds=point_clouds,
    homology_dimension=homology_dimension,
    n_jobs=n_jobs,
)

    records = []

    for i, (_, row) in enumerate(asset_windows_df.iterrows()):
        diagram = diagrams[i]

        tp_h1, n_h1_features = compute_tp_h1_from_diagram(
            diagram=diagram,
            homology_dimension=homology_dimension,
        )

        records.append(
            {
                "asset": asset,
                "window_id": int(row["window_id"]),
                "reference_date": row["reference_date"],
                "window_start_date": row["window_start_date"],
                "window_end_date": row["window_end_date"],
                "tau": tau,
                "embedding_dimension": dimension,
                "n_window_obs": int(row["n_obs"]),
                "n_embedding_points": int(n_embedding_points_list[i]),
                "homology_dimension": homology_dimension,
                "tp_h1": tp_h1,
                "n_h1_features": n_h1_features,
            }
        )

    return pd.DataFrame(records)


def compute_real_topology(
    windows_df,
    metadata_df,
    parameters_df,
    homology_dimension=1,
    n_jobs=-1,
):
    """
Calcula TP_H1 real para todos os ativos e retorna uma tabela por janela.

    """
    merged_df = prepare_windows_with_metadata(
        windows_df=windows_df,
        metadata_df=metadata_df,
    )

    result_dfs = []

    assets = merged_df["asset"].unique()

    for asset in tqdm(assets, desc="Calculando topologia real por ativo"):
        asset_windows_df = merged_df[merged_df["asset"] == asset].copy()

        asset_result = compute_topology_for_asset(
            asset_windows_df=asset_windows_df,
            parameters_df=parameters_df,
            asset=asset,
            homology_dimension=homology_dimension,
            n_jobs=n_jobs,
        )

        result_dfs.append(asset_result)

    results_df = pd.concat(result_dfs, ignore_index=True)

    return results_df


def validate_topology_results(results_df):
    """
    Valida a consistência básica da tabela de resultados topológicos.
    """
    if results_df.empty:
        raise ValueError("A tabela de resultados topológicos está vazia.")

    required_columns = [
        "asset",
        "window_id",
        "reference_date",
        "tau",
        "embedding_dimension",
        "tp_h1",
        "n_h1_features",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in results_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Colunas ausentes nos resultados: {missing_columns}")

    if results_df["tp_h1"].isna().any():
        raise ValueError("Há valores NaN em tp_h1.")

    if np.isinf(results_df["tp_h1"].to_numpy()).any():
        raise ValueError("Há valores infinitos em tp_h1.")

    if (results_df["tp_h1"] < 0).any():
        raise ValueError("Há valores negativos em tp_h1.")


def compute_persistence_diagrams(point_clouds, homology_dimension=1, n_jobs=-1):
    """
    Calcula diagramas de persistência para uma ou mais nuvens de pontos.
    """
    point_clouds = np.asarray(point_clouds, dtype=float)

    persistence = VietorisRipsPersistence(
        homology_dimensions=[homology_dimension],
        n_jobs=n_jobs,
    )

    return persistence.fit_transform(point_clouds)

