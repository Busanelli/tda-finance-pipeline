from pathlib import Path
from datetime import datetime

import yaml


def load_config(config_path="config.yaml"):
    """
    Carrega o arquivo config.yaml e retorna um dicionário de configuração.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def get_project_root():
    """
    Retorna a raiz do projeto.

    Assume que este arquivo está dentro da pasta src/.
    """
    return Path(__file__).resolve().parents[1]


def get_path(config, path_key):
    """
    Retorna um caminho definido na seção paths do config.yaml.
    """
    project_root = get_project_root()

    try:
        relative_path = config["paths"][path_key]
    except KeyError as error:
        raise KeyError(
            f"Caminho não encontrado em config['paths']: {path_key}"
        ) from error

    return project_root / relative_path


def ensure_directories(config):
    """
    Cria, se necessário, todos os diretórios definidos em config['paths'].
    """
    for _, relative_path in config["paths"].items():
        full_path = get_project_root() / relative_path
        full_path.mkdir(parents=True, exist_ok=True)


def validate_config(config):
    """
    Faz validações básicas no dicionário de configuração.
    """
    required_sections = [
        "project",
        "data",
        "windowing",
        "parameters",
        "topology",
        "surrogates",
        "paths",
    ]

    for section in required_sections:
        if section not in config:
            raise KeyError(
                f"Seção obrigatória ausente no config.yaml: {section}"
            )

    if "assets" not in config["data"]:
        raise KeyError("Chave obrigatória ausente: data.assets")

    if len(config["data"]["assets"]) == 0:
        raise ValueError("A lista de ativos está vazia.")

    if config["windowing"]["window_size"] <= 0:
        raise ValueError("window_size deve ser maior que zero.")

    if config["windowing"]["step_size"] <= 0:
        raise ValueError("step_size deve ser maior que zero.")

    if config["surrogates"]["n_surrogates"] <= 0:
        raise ValueError("n_surrogates deve ser maior que zero.")


def setup_project(config_path="config.yaml"):
    """
    Carrega, valida e prepara a estrutura de diretórios do projeto.
    """
    config = load_config(config_path)
    validate_config(config)
    ensure_directories(config)

    return config

def log_message(message):
    """
    Exibe uma mensagem com timestamp no terminal.
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] {message}", flush=True)