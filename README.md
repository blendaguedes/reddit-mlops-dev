# Reddit MLOps — DEV

Projeto de classificacao de sentimento/engajamento em posts do Reddit, construido como um pipeline de Machine Learning com boas praticas de MLOps.

---

## O que este projeto faz

Dado um post do Reddit, o modelo prediz se ele vai ter **alto ou baixo engajamento** (score acima ou abaixo da mediana), usando apenas o texto do post como feature.

**Pipeline:**
```
Kaggle Dataset → Limpeza de Texto → TF-IDF → Classificador → Metricas
```

---

## Tecnologias utilizadas

| Ferramenta | Para que serve |
|---|---|
| **Python 3.11** | Linguagem principal |
| **pandas / numpy** | Manipulacao de dados |
| **scikit-learn** | Preprocessamento e modelos de ML |
| **scipy** | Matrizes esparsas (TF-IDF) |
| **kagglehub** | Download automatico do dataset do Kaggle |
| **MLflow** | Rastreamento de experimentos (metricas, parametros, modelos) |
| **DVC** | Versionamento de dados e pipeline reproduzivel |
| **joblib** | Serializacao de modelos |
| **python-dotenv** | Gerenciamento de variaveis de ambiente |
| **Jupyter** | Notebooks de experimentacao e analise |
| **matplotlib / seaborn** | Visualizacoes |

---

## Estrutura do projeto

```
reddit-mlops-dev/
│
├── src/                        # Modulos reutilizaveis do projeto
│   ├── config.py               # Configuracoes globais e caminhos
│   ├── data_loader.py          # Download e carregamento do dataset
│   ├── preprocessor.py         # Limpeza de texto e vetorizacao TF-IDF
│   └── model_trainer.py        # Treinamento e avaliacao dos modelos
│
├── pipeline/                   # Scripts das stages do DVC
│   ├── load.py                 # Stage 1: baixa e salva dados brutos
│   ├── preprocess.py           # Stage 2: processa e salva artefatos
│   └── train.py                # Stage 3: treina modelos e salva metricas
│
├── notebooks/                  # Experimentacao (rode antes de fixar parametros)
│   ├── 01_data_analysis.ipynb  # EDA: entender os dados
│   ├── 02_preprocessing_analysis.ipynb  # Escolher parametros TF-IDF
│   └── 03_model_selection.ipynb         # Comparar modelos
│
├── data/
│   ├── raw/                    # Dados brutos (reddit_data.csv) — DVC rastreia
│   └── processed/              # Artefatos processados (X.npz, y.npy, ...) — DVC rastreia
│
├── models/                     # Modelos treinados (.joblib) — DVC rastreia
├── metrics/                    # Metricas do pipeline (scores.json)
├── logs/                       # Logs de execucao
│
├── main.py                     # Runner standalone (sem DVC, para testes rapidos)
├── dvc.yaml                    # Definicao das stages do pipeline DVC
├── dvc.lock                    # Lock file do DVC (versionado pelo git)
├── params.yaml                 # Hiperparametros do pipeline
├── requirements.txt            # Dependencias Python
├── Makefile                    # Atalhos de comandos
├── .env                        # Credenciais locais (NAO versionar)
└── .env.example                # Exemplo de variaveis de ambiente
```

---

## Arquivos de configuracao e infraestrutura

Esta secao explica o proposito de cada arquivo nao-Python do projeto. Entender esses arquivos e fundamental para trabalhar com MLOps.

---

### `params.yaml`

Arquivo central de hiperparametros e configuracoes do pipeline. Todos os valores que afetam o comportamento do modelo ou do preprocessamento ficam aqui — nunca hardcoded no codigo.

```yaml
preprocess:
  max_features: 5000   # tamanho do vocabulario TF-IDF
  min_df: 5            # minimo de documentos para uma palavra entrar no vocabulario
train:
  test_size: 0.2       # proporcao do conjunto de teste
```

O DVC monitora este arquivo. Se voce alterar qualquer valor, na proxima vez que rodar `dvc repro` ele re-executa automaticamente apenas as stages afetadas pela mudanca.

---

### `dvc.yaml`

Define as stages (etapas) do pipeline de dados. E o equivalente de um `Makefile` para dados e modelos: diz ao DVC o que executar, do que cada stage depende e o que ela produz.

Estrutura de cada stage:
- `cmd`: o comando a executar
- `deps`: arquivos e codigos dos quais esta stage depende (se mudar, a stage re-executa)
- `params`: parametros do `params.yaml` que esta stage usa
- `outs`: arquivos que esta stage produz e que o DVC deve rastrear
- `metrics`: arquivos de metricas que podem ser comparados entre runs

---

### `dvc.lock`

Arquivo gerado automaticamente pelo DVC apos cada execucao. Registra os hashes (MD5) de todos os arquivos de entrada e saida de cada stage no momento em que rodaram. Serve para garantir reproducibilidade: com este arquivo, qualquer pessoa consegue verificar se o ambiente esta identico ao original.

Deve ser versionado pelo git — e o "recibo" de uma execucao do pipeline.

---

### `.dvcignore`

Equivalente ao `.gitignore`, mas para o DVC. Lista arquivos e pastas que o DVC deve ignorar ao calcular hashes e rastrear mudancas. Criado automaticamente pelo `dvc init`.

---

### `requirements.txt`

Lista todas as dependencias Python do projeto com suas versoes fixadas. Garante que qualquer pessoa que instalar o projeto com `pip install -r requirements.txt` obtenha exatamente o mesmo ambiente.

Boas praticas:
- Fixe versoes (`scikit-learn==1.3.2`) para garantir reproducibilidade
- Separe dependencias de producao das de desenvolvimento se o projeto crescer

---

### `Makefile`

Centraliza os comandos mais usados do projeto em atalhos simples. Em vez de decorar `python -m dvc repro` ou `python -m mlflow ui`, basta usar `make pipeline` ou `make mlflow`.

Requer o programa `make` instalado no sistema (nativo em Linux/Mac; no Windows instalar via `winget install GnuWin32.Make`).

---

### `.env`

Arquivo de variaveis de ambiente locais — credenciais, chaves de API e configuracoes especificas da maquina. Nunca deve ser versionado no git (ja esta no `.gitignore`).

Cada desenvolvedor tem o seu proprio `.env` com suas credenciais. O arquivo `.env.example` serve de guia para saber quais variaveis precisam ser preenchidas.

---

### `.env.example`

Template do `.env` com os nomes das variaveis necessarias, mas sem valores reais. Versionado no git para que novos desenvolvedores saibam o que precisam configurar.

Fluxo de uso:
```bash
cp .env.example .env
# editar .env com suas credenciais reais
```

---

### `.gitignore`

Diz ao git quais arquivos e pastas nao devem ser versionados. Neste projeto, os principais ignorados sao:
- `.venv/` — ambiente virtual (pesado, recriavel com `pip install`)
- `data/` — dados rastreados pelo DVC, nao pelo git
- `models/` — modelos binarios rastreados pelo DVC
- `.env` — credenciais locais
- `__pycache__/` — cache do Python, gerado automaticamente

---

### `.dvc/config`

Arquivo de configuracao do repositorio DVC. Armazena configuracoes como o endereco do remote storage (S3, GCS, DagsHub) quando configurado. Versionado pelo git para que toda a equipe compartilhe as mesmas configuracoes de DVC.

---

## Como construir um projeto como este — passo a passo

Esta secao explica o **raciocinio por tras de cada decisao** de construcao.

### Passo 1 — Defina o problema e o dataset

Antes de escrever qualquer codigo, responda:
- Qual e a tarefa? (classificacao, regressao, clustering...)
- Qual e a variavel alvo (y)?
- Qual e a fonte dos dados?

Neste projeto: classificacao binaria de engajamento usando o dataset `pavellexyr/the-reddit-dataset-dataset` do Kaggle.

### Passo 2 — Crie a estrutura de pastas

Separe responsabilidades desde o inicio:

```
projeto/
├── src/          # logica reutilizavel
├── pipeline/     # scripts de execucao
├── notebooks/    # experimentacao
├── data/         # dados (nunca versionar no git)
├── models/       # modelos serializados
└── metrics/      # resultados
```

> **Por que separar src/ de pipeline/?**
> `src/` contem classes e funcoes reutilizaveis que podem ser importadas em qualquer lugar (notebooks, testes, scripts).
> `pipeline/` contem scripts de execucao que o DVC chama — cada um faz uma coisa so e persiste o resultado em disco.

### Passo 3 — Explore os dados nos notebooks ANTES de escrever o pipeline

A ordem correta e:
1. **Notebook 01** (EDA): entenda o dataset — colunas, nulos, distribuicao do target
2. **Notebook 02** (Preprocessing): experimente parametros de limpeza e vetorizacao
3. **Notebook 03** (Model Selection): compare modelos e escolha o melhor

So depois disso fixe os parametros no `params.yaml` e escreva o pipeline.

> **Por que notebooks primeiro?**
> Mudar parametros num notebook e instantaneo. Mudar depois que o pipeline esta pronto e mais custoso e arriscado.

### Passo 4 — Escreva os modulos src/

Cada modulo tem uma responsabilidade:

- `config.py`: centraliza caminhos e parametros. Nunca hardcode paths em outros arquivos
- `data_loader.py`: isola a logica de aquisicao de dados. Se a fonte mudar, so este arquivo muda
- `preprocessor.py`: salva todos os artefatos em disco (`X.npz`, `y.npy`, `vectorizer.joblib`). Nenhum dado deve existir apenas em RAM
- `model_trainer.py`: treina, avalia e loga no MLflow. Retorna metricas estruturadas

### Passo 5 — Configure o versionamento com DVC

DVC resolve dois problemas:
1. **Dados grandes** nao podem ir pro git — DVC rastreia os arquivos e guarda apenas hashes
2. **Reproducibilidade** — se um parametro ou dependencia muda, DVC sabe exatamente quais stages precisam rodar novamente

Estrutura do `dvc.yaml`:
```yaml
stages:
  nome_da_stage:
    cmd: python pipeline/script.py   # comando a executar
    deps:                            # o que esta stage depende
      - pipeline/script.py
      - src/modulo.py
      - data/entrada/arquivo.csv
    params:                          # parametros do params.yaml que afetam esta stage
      - secao.parametro
    outs:                            # o que esta stage produz (DVC rastreia)
      - data/saida/arquivo.npz
    metrics:                         # arquivos de metricas (opcional)
      - metrics/scores.json:
          cache: false
```

### Passo 6 — Rastreie experimentos com MLflow

MLflow registra automaticamente para cada run:
- Parametros do modelo (`mlflow.log_params`)
- Metricas de avaliacao (`mlflow.log_metrics`)
- O modelo serializado (`mlflow.sklearn.log_model`)

Isso permite comparar runs diferentes na UI web.

### Passo 7 — Centralize parametros no params.yaml

Nunca hardcode valores como `max_features=5000` diretamente no codigo.
Coloque em `params.yaml` e leia com `yaml.safe_load()`.

Beneficio: quando voce muda um parametro, o DVC detecta a mudanca e re-executa apenas as stages afetadas.

### Passo 8 — Automatize com Makefile

Um `Makefile` torna o projeto acessivel para qualquer pessoa com um unico comando:
```bash
make pipeline   # roda tudo
make metrics    # ve os resultados
make clean      # limpa artefatos
```

---

## Como executar o projeto

### Pre-requisitos

- Python 3.11+
- Git
- Conta no [Kaggle](https://www.kaggle.com) com API token

### 1. Clone o repositorio

```bash
git clone <url-do-repositorio>
cd reddit-mlops-dev
```

### 2. Crie e ative o ambiente virtual

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / Mac
python -m venv .venv
source .venv/bin/activate
```

### 3. Instale as dependencias

```bash
pip install -r requirements.txt
```

### 4. Configure as credenciais do Kaggle

1. Acesse [kaggle.com/settings/account](https://www.kaggle.com/settings/account)
2. Clique em **Create New API Token** — faz download do `kaggle.json`
3. Coloque o arquivo em:
   - **Windows:** `C:\Users\SEU_USUARIO\.kaggle\kaggle.json`
   - **Linux/Mac:** `~/.kaggle/kaggle.json`

### 5. Configure o arquivo .env

```bash
cp .env.example .env
```

Edite o `.env` com suas credenciais:
```
KAGGLE_USERNAME=seu_username
KAGGLE_KEY=sua_api_key
```

### 6. Execute o pipeline DVC

```bash
# Roda todas as stages em sequencia
python -m dvc repro
```

O DVC executa automaticamente apenas as stages que precisam rodar (detecta mudancas em deps, params e outs).

**Ou use o Makefile (requer `make` instalado):**
```bash
make pipeline
```

### 7. Visualize as metricas

```bash
# Via DVC
python -m dvc metrics show

# Via MLflow UI (abra http://localhost:5000)
python -m mlflow ui
```

---

## Executando stages individualmente

```bash
# Apenas download dos dados
python -m dvc repro load

# Apenas preprocessamento
python -m dvc repro preprocess

# Apenas treinamento
python -m dvc repro train
```

---

## Executando sem DVC (modo standalone)

Para testes rapidos sem passar pelo DVC:

```bash
python main.py
```

---

## Explorando os notebooks

Inicie o Jupyter Lab e abra a pasta `notebooks/`:

```bash
python -m jupyter lab
```

| Notebook | O que explorar |
|---|---|
| `01_data_analysis.ipynb` | Distribuicao dos dados, nulos, target |
| `02_preprocessing_analysis.ipynb` | Impacto dos parametros TF-IDF |
| `03_model_selection.ipynb` | Comparacao de modelos, feature importance |

---

## Alterando parametros

Edite o `params.yaml` e rode `dvc repro` — o DVC detecta a mudanca e re-executa apenas as stages afetadas:

```yaml
preprocess:
  max_features: 5000   # tente 2000 ou 10000
  max_df: 0.8
  min_df: 5            # tente 2 ou 10

train:
  test_size: 0.2       # tente 0.15 ou 0.25
  random_state: 42
```

---

## Resultados obtidos

Metricas no conjunto de teste (80/20 split, 54.061 amostras):

| Modelo | F1 | AUC | Tempo de treino |
|---|---|---|---|
| Logistic Regression | 0.4233 | 0.6231 | 0.3s |
| Naive Bayes | 0.4024 | 0.6249 | ~0s |
| Random Forest | 0.3791 | 0.6194 | 95.8s |

**Modelo escolhido:** Logistic Regression — melhor F1, treino rapido, interpretavel.

> A performance moderada (AUC ~0.62) e esperada: o score de um post no Reddit depende de fatores contextuais (horario, autor, subreddit) que nao estao no texto. Isso e um resultado valido e documentado nos notebooks.

---

## Comandos uteis

```bash
# Ver o grafo do pipeline
python -m dvc dag

# Ver status (o que mudou e precisa re-rodar)
python -m dvc status

# Limpar artefatos processados e modelos
make clean          # ou: rm -f data/processed/* models/* metrics/scores.json

# Limpar tudo incluindo dados brutos (proximo repro vai re-baixar do Kaggle)
make clean-all
```

---

## Proximos passos sugeridos

Para evoluir este projeto:

1. **Adicionar features de metadata** — comprimento do texto, hora de criacao, coluna `sentiment`
2. **Tuning de hiperparametros** — `GridSearchCV` ou `Optuna` para LR e RF
3. **Testes automatizados** — `pytest` para validar preprocessamento e modelos
4. **CI/CD** — GitHub Actions para rodar `dvc repro` automaticamente em cada push
5. **Remote storage para DVC** — S3, GCS ou DagsHub para compartilhar dados da equipe
6. **Evoluir para o projeto PROD** — containerizacao com Docker, API com FastAPI, Kubernetes
