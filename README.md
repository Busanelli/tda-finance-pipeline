# TDA Finance Pipeline

Pipeline em Python para análise de séries temporais financeiras com **Topological Data Analysis (TDA)** e **Homologia Persistente**.

O projeto transforma janelas de retornos financeiros em nuvens de pontos por embedding de Takens e calcula a persistência total em dimensão $H_1$. As séries topológicas resultantes são comparadas com volatilidade móvel e com envelopes empíricos gerados por surrogates.

A aplicação padrão utiliza ativos da B3, mas o pipeline foi organizado de forma reprodutível e configurável, permitindo novos experimentos com outros ativos, períodos e parâmetros.

## Contexto

Este repositório foi desenvolvido a partir do Trabalho de Conclusão de Curso:

**Persistência Total em Homologia Persistente como Medida Complementar para Análise de Séries Temporais Financeiras**

O texto completo do TCC está disponível em [docs/tcc-tda-series-financeiras.pdf](docs/tcc-tda-series-financeiras.pdf) para consulta da fundamentação teórica, metodologia e interpretação dos resultados.

## Saídas do pipeline

A execução padrão gera saídas consolidadas em `outputs/`, organizadas em tabelas e figuras.

Entre os artefatos gerados estão tabelas por janela, resumos por ativo, matrizes de correlação entre séries topológicas, comparações com volatilidade móvel, envelopes empíricos baseados em surrogates e diagramas de persistência representativos.

## Configuração

Os principais parâmetros do experimento ficam em `config.yaml`, incluindo ativos analisados, período de coleta, tamanho e passo das janelas móveis, parâmetros de estimação de embedding, dimensão de homologia, número de surrogates e caminhos de entrada e saída.

Isso permite adaptar o pipeline para novos experimentos sem alterar diretamente o código-fonte.

## Como executar

Com as dependências instaladas, o pipeline completo pode ser executado com:

```bash
python run_pipeline.py
``` 
O arquivo `run_pipeline.py` executa os scripts do diretório `scripts/` de forma sequencial. As etapas também podem ser executadas individualmente, desde que as dependências entre elas sejam respeitadas.

Durante a execução, o pipeline gera arquivos intermediários em diretórios como `data/`, `tables/` e `figures/`. Esses arquivos são regeneráveis e não estão versionados no repositório.

## Estrutura do projeto

```text
tda-finance-pipeline/
├── config.yaml
├── README.md
├── requirements.txt
├── run_pipeline.py
├── scripts/
├── src/
└── outputs/
```

## Tecnologias e bibliotecas

- Python 3
- pandas
- NumPy
- SciPy
- scikit-learn
- statsmodels
- yfinance
- giotto-tda
- matplotlib
- seaborn
- PyYAML
- tqdm

As dependências estão listadas em `requirements.txt` e podem ser instaladas com:

```bash
pip install -r requirements.txt
```