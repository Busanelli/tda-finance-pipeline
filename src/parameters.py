import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import acf
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.neighbors import NearestNeighbors

def standardize_values(values):
    """
 Padroniza uma série por z-score, usada apenas na estimação dos parâmetros.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        raise ValueError("Série vazia após remoção de NaN.")

    std = values.std(ddof=0)

    if std == 0:
        raise ValueError("Série com desvio-padrão zero.")

    return (values - values.mean()) / std


def compute_ami_curve(values, max_lag, n_bins=10):
    """
    Calcula a curva de informação mútua aproximada para diferentes lags.

    Antes do cálculo, a série contínua é discretizada em bins por quantis.
    """
    values = standardize_values(values)

    discretizer = KBinsDiscretizer(
        n_bins=n_bins,
        encode="ordinal",
        strategy="quantile",
    )

    discrete_values = discretizer.fit_transform(
        values.reshape(-1, 1)
    ).astype(int).ravel()

    records = []

    for lag in range(1, max_lag + 1):
        x = discrete_values[:-lag]
        y = discrete_values[lag:]

        ami = mutual_info_score(x, y)

        records.append(
            {
                "lag": lag,
                "ami": ami,
            }
        )

    return pd.DataFrame(records)


def find_first_local_minimum(curve_df, value_column):
    """
Retorna o primeiro mínimo local de uma curva.

Caso não exista mínimo local, usa o mínimo global como fallback.
    """
    values = curve_df[value_column].to_numpy()
    lags = curve_df["lag"].to_numpy()

    for i in range(1, len(values) - 1):
        previous_value = values[i - 1]
        current_value = values[i]
        next_value = values[i + 1]

        if current_value < previous_value and current_value < next_value:
            return int(lags[i]), "first_local_minimum"

    min_index = int(np.argmin(values))

    return int(lags[min_index]), "global_minimum_fallback"


def compute_acf_curve(values, max_lag):
    """
    Calcula a função de autocorrelação (ACF) para diferentes lags.
    """
    values = standardize_values(values)

    acf_values = acf(
        values,
        nlags=max_lag,
        fft=True,
        missing="raise",
    )

    records = []

    for lag in range(1, max_lag + 1):
        records.append(
            {
                "lag": lag,
                "acf": acf_values[lag],
            }
        )

    return pd.DataFrame(records)


def find_first_acf_below_threshold(acf_df, threshold=1 / np.e):
    """
    Retorna o primeiro lag em que a ACF fica abaixo do limiar.

    Caso nenhum lag satisfaça o critério, usa o menor valor de ACF como fallback.
    """
    below_threshold = acf_df[acf_df["acf"] <= threshold]

    if not below_threshold.empty:
        lag = below_threshold.iloc[0]["lag"]
        return int(lag), "first_below_threshold"

    min_index = acf_df["acf"].idxmin()
    lag = acf_df.loc[min_index, "lag"]

    return int(lag), "minimum_acf_fallback"


def takens_embedding(values, tau, dimension):
    """
    Cria uma matriz de embedding de Takens.

    Cada linha contém os valores [x_t, x_{t+tau}, ..., x_{t+(m-1)tau}].
    """
    values = np.asarray(values, dtype=float)

    n_vectors = len(values) - (dimension - 1) * tau

    if n_vectors <= 1:
        raise ValueError(
            "Série curta demais para o tau e a dimensão informados."
        )

    embedded = np.empty((n_vectors, dimension))

    for j in range(dimension):
        embedded[:, j] = values[j * tau: j * tau + n_vectors]

    return embedded


def false_nearest_neighbors_ratio(
    values,
    tau,
    dimension,
    rtol=10.0,
    atol=2.0,
):
    """
    Calcula a proporção de falsos vizinhos para uma dimensão.

    Compara os embeddings nas dimensões m e m+1.
    """
    values = standardize_values(values)

    embedding_m = takens_embedding(values, tau=tau, dimension=dimension)
    embedding_m1 = takens_embedding(values, tau=tau, dimension=dimension + 1)

    n_valid = embedding_m1.shape[0]
    embedding_m = embedding_m[:n_valid]

    if n_valid < 3:
        raise ValueError("Poucos pontos para calcular FNN.")

    nearest_neighbors = NearestNeighbors(n_neighbors=2)
    nearest_neighbors.fit(embedding_m)

    distances, indices = nearest_neighbors.kneighbors(embedding_m)

    nearest_distance_m = distances[:, 1]
    nearest_index = indices[:, 1]

    extra_coordinate_current = values[dimension * tau: dimension * tau + n_valid]
    extra_coordinate_neighbor = extra_coordinate_current[nearest_index]

    extra_distance = np.abs(
        extra_coordinate_current - extra_coordinate_neighbor
    )

    epsilon = 1e-12
    ratio_test = extra_distance / (nearest_distance_m + epsilon)

    distance_m1 = np.sqrt(nearest_distance_m ** 2 + extra_distance ** 2)

    false_neighbors = (
        (ratio_test > rtol)
        | (distance_m1 > atol)
    )

    return false_neighbors.mean()


def compute_fnn_curve(
    values,
    tau,
    min_dimension,
    max_dimension,
    threshold,
    rtol=10.0,
    atol=2.0,
):
    """
    Calcula a curva de falsos vizinhos (FNN) para diferentes dimensões de embedding.
    """
    records = []

    for dimension in range(min_dimension, max_dimension + 1):
        fnn_ratio = false_nearest_neighbors_ratio(
            values=values,
            tau=tau,
            dimension=dimension,
            rtol=rtol,
            atol=atol,
        )

        records.append(
            {
                "dimension": dimension,
                "fnn_ratio": fnn_ratio,
                "threshold": threshold,
            }
        )

    return pd.DataFrame(records)


def select_embedding_dimension(fnn_df, threshold):
    """
    Seleciona a dimensão de embedding a partir da curva FNN.

    Usa a primeira dimensão abaixo do limiar. Se o limiar não for atingido,
    usa o primeiro mínimo local; se necessário, recorre ao mínimo global.
    """
    below_threshold = fnn_df[fnn_df["fnn_ratio"] <= threshold]

    if not below_threshold.empty:
        dimension = below_threshold.iloc[0]["dimension"]
        return int(dimension), "first_below_threshold"

    values = fnn_df["fnn_ratio"].to_numpy()
    dimensions = fnn_df["dimension"].to_numpy()

    for i in range(1, len(values) - 1):
        previous_value = values[i - 1]
        current_value = values[i]
        next_value = values[i + 1]

        if current_value < previous_value and current_value < next_value:
            return int(dimensions[i]), "first_local_minimum"

    min_index = fnn_df["fnn_ratio"].idxmin()
    dimension = fnn_df.loc[min_index, "dimension"]

    return int(dimension), "global_minimum_fallback"


def estimate_parameters_for_asset(
    series,
    asset,
    max_lag,
    min_dimension,
    max_dimension,
    fnn_threshold,
    n_bins=10,
):
    """
    Estima tau e m para um ativo.

    Retorna os parâmetros finais e as curvas AMI, ACF e FNN usadas na estimação.
    """
    values = standardize_values(series.to_numpy(dtype=float))

    ami_df = compute_ami_curve(
        values=values,
        max_lag=max_lag,
        n_bins=n_bins,
    )
    tau_ami, tau_ami_rule = find_first_local_minimum(
        curve_df=ami_df,
        value_column="ami",
    )

    acf_df = compute_acf_curve(
        values=values,
        max_lag=max_lag,
    )
    tau_acf, tau_acf_rule = find_first_acf_below_threshold(acf_df)

    tau_final = tau_ami
    tau_final_rule = "ami_first_local_minimum"

    fnn_df = compute_fnn_curve(
        values=values,
        tau=tau_final,
        min_dimension=min_dimension,
        max_dimension=max_dimension,
        threshold=fnn_threshold,
    )
    m_fnn, m_fnn_rule = select_embedding_dimension(
        fnn_df=fnn_df,
        threshold=fnn_threshold,
    )

    m_final = m_fnn
    m_final_rule = "fnn_first_below_threshold"

    ami_df.insert(0, "asset", asset)
    acf_df.insert(0, "asset", asset)
    fnn_df.insert(0, "asset", asset)

    parameter_record = {
        "asset": asset,
        "n_obs": len(values),

        "tau_ami": tau_ami,
        "tau_ami_rule": tau_ami_rule,

        "tau_acf": tau_acf,
        "tau_acf_rule": tau_acf_rule,

        "tau_final": tau_final,
        "tau_final_rule": tau_final_rule,

        "m_fnn": m_fnn,
        "m_fnn_rule": m_fnn_rule,

        "m_final": m_final,
        "m_final_rule": m_final_rule,

        "max_lag": max_lag,
        "min_dimension": min_dimension,
        "max_dimension": max_dimension,
        "fnn_threshold": fnn_threshold,
        "ami_n_bins": n_bins,
    }

    return parameter_record, ami_df, acf_df, fnn_df


def estimate_parameters_for_all_assets(log_returns, config):
    """
    Estima tau e m para todos os ativos e reúne as curvas AMI, ACF e FNN.
    """
    tau_config = config["parameters"]["tau"]
    dimension_config = config["parameters"]["embedding_dimension"]

    max_lag = tau_config.get("max_lag", 30)
    n_bins = tau_config.get("n_bins", 10)

    min_dimension = dimension_config.get("min_dimension", 2)
    max_dimension = dimension_config.get("max_dimension", 10)
    fnn_threshold = dimension_config.get("threshold", 0.01)

    parameter_records = []
    ami_dfs = []
    acf_dfs = []
    fnn_dfs = []

    for asset in log_returns.columns:
        parameter_record, ami_df, acf_df, fnn_df = estimate_parameters_for_asset(
            series=log_returns[asset],
            asset=asset,
            max_lag=max_lag,
            min_dimension=min_dimension,
            max_dimension=max_dimension,
            fnn_threshold=fnn_threshold,
            n_bins=n_bins,
        )

        parameter_records.append(parameter_record)
        ami_dfs.append(ami_df)
        acf_dfs.append(acf_df)
        fnn_dfs.append(fnn_df)

    parameters_df = pd.DataFrame(parameter_records)
    ami_values_df = pd.concat(ami_dfs, ignore_index=True)
    acf_values_df = pd.concat(acf_dfs, ignore_index=True)
    fnn_values_df = pd.concat(fnn_dfs, ignore_index=True)

    return parameters_df, ami_values_df, acf_values_df, fnn_values_df

