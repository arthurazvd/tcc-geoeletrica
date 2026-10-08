# Inversão Geoelétrica 1D Baseada em Redes Neurais

Projeto de pesquisa/TCC que investiga o uso de **redes neurais artificiais do tipo MLP (Multi-Layer Perceptron)** para estimar parâmetros de modelos geoelétricos unidimensionais a partir de curvas sintéticas de **Sondagem Elétrica Vertical (SEV)**, utilizando o arranjo Schlumberger.

## Objetivo

A inversão geoelétrica busca recuperar propriedades do subsolo a partir de medidas de resistividade aparente. Neste projeto, uma rede neural aprende a relação entre uma curva de SEV e cinco parâmetros de um modelo de três camadas:

| Parâmetro | Significado | Unidade |
| --- | --- | --- |
| `rho1`, `rho2`, `rho3` | Resistividades elétricas das três camadas | Ω·m |
| `h1`, `h2` | Espessuras das duas primeiras camadas | m |

A terceira camada é considerada um semiespaço. A entrada da rede contém **20 valores de resistividade aparente**, correspondentes a espaçamentos `AB/2` entre 1 e 200 m.

## O que o projeto faz

1. **Modelagem direta:** calcula curvas de resistividade aparente para modelos de três camadas com integração numérica e função de Bessel `J1`.
2. **Geração de dados:** cria conjuntos sintéticos de modelos geoelétricos e suas curvas, exportados em CSV.
3. **Treinamento:** treina uma MLP para estimar os cinco parâmetros a partir dos 20 valores da curva.
4. **Predição:** compara parâmetros conhecidos de um modelo sintético com as estimativas da rede.
5. **Experimentos com ruído:** treina e compara modelos independentes para níveis de ruído de 0%, 2% e 5%.
6. **Avaliação:** gera métricas, tabelas e gráficos para análise dos resultados.

> **Escopo:** o projeto trabalha com dados sintéticos. A validação com dados reais de campo não está demonstrada nesta versão.

## Tecnologias

- Python 3
- NumPy e SciPy: operações numéricas e modelagem direta
- Pandas: leitura e manipulação de conjuntos de dados
- scikit-learn: treinamento e avaliação da MLP
- Matplotlib: gráficos
- Joblib: armazenamento dos modelos treinados

## Estrutura do repositório

```text
tcc-geoeletrica/
├── data/                     # Bases sintéticas em CSV
├── models/                   # Modelos treinados (.joblib)
│   └── noise/                # Modelos por nível de ruído
├── results/
│   ├── figures/              # Gráficos produzidos
│   ├── metrics/              # Métricas do treinamento principal
│   └── noise/                # Métricas e previsões com ruído
├── src/
│   ├── forward_model.py      # Modelagem direta da SEV
│   ├── models.py             # Representação dos modelos geoelétricos
│   ├── generate_dataset.py   # Geração do dataset
│   ├── ml_pipeline.py        # Pipeline de aprendizado de máquina
│   ├── train_mlp.py          # Treinamento principal
│   ├── predict.py            # Predição com a MLP principal
│   ├── noise_experiments.py  # Experimentos com ruído
│   ├── predict_noise.py      # Predição usando as três redes
│   └── demo.py               # Demonstração da modelagem direta
├── tests/
│   └── test_forward.py       # Teste do modelo direto
├── .gitignore
├── requirements.txt
└── README.md
```

## Instalação

É recomendável utilizar Python 3.11 ou superior e executar os comandos **na raiz do projeto**.

### 1. Criar e ativar o ambiente virtual

**Windows (PowerShell):**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Se a ativação estiver bloqueada no PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar as dependências

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

O `.gitignore` impede que a pasta `.venv/` seja incluída em novos commits. Cada pessoa pode criar seu próprio ambiente virtual.

## Como executar

### Etapa 1: modelagem direta e geração de dados

Execute a demonstração de uma curva de SEV:

```bash
python -m src.demo
```

O gráfico é salvo em `results/figures/demo_sev.png`.

Para gerar uma base pequena de teste:

```bash
python -m src.generate_dataset --n 100 --output data/exemplo_100.csv
```

Para gerar a base de 30.000 modelos:

```bash
python -m src.generate_dataset --n 30000 --du 0.0005 --output data/dataset_30000.csv
```

O parâmetro `--du` controla o passo da integração numérica. O valor `0.0005` exige mais processamento que o padrão `0.002`.

**Formato do CSV:** 20 colunas `rhoa_01` a `rhoa_20` (entradas) e cinco colunas `rho1`, `rho2`, `rho3`, `h1`, `h2` (alvos). As resistividades dos modelos sintéticos são amostradas entre 10 e 1000 Ω·m; `h1` entre 1 e 20 m; `h2` entre 2 e 50 m.

### Etapa 2: treinamento e predição da MLP

O ZIP já contém `data/dataset_30000.csv`, então não é necessário regenerá-lo para começar a treinar.

```bash
python -m src.train_mlp --dataset data/dataset_30000.csv
```

Esse comando salva `models/mlp_bundle.joblib` e arquivos de avaliação em `results/metrics/`.

Para testar a inversão com um exemplo sintético:

```bash
python -m src.predict
```

Para informar parâmetros específicos:

```bash
python -m src.predict --rho1 80 --rho2 300 --rho3 40 --h1 8 --h2 25
```

O programa compara os parâmetros reais com os estimados e salva `results/figures/prediction_demo.png`.

### Etapa 3: robustez a ruído

Treine e compare as redes para níveis de ruído de **0%, 2% e 5%**:

```bash
python -m src.noise_experiments --dataset data/dataset_30000.csv
```

Para também avaliar o erro das curvas previstas (processamento adicional):

```bash
python -m src.noise_experiments --dataset data/dataset_30000.csv --curve-metrics --du 0.0005
```

Os modelos ficam em `models/noise/`; as métricas em `results/noise/`; o gráfico comparativo em `results/figures/noise_parameter_errors.png`.

Compare as previsões dos três modelos em um mesmo terreno:

```bash
python -m src.predict_noise --rho1 80 --rho2 300 --rho3 40 --h1 8 --h2 25
```

Os valores estimados e os erros dependem do treinamento, das sementes e das versões das bibliotecas. Não são resultados fixos.

## Testes

Há um teste básico do modelo direto para o caso de meio homogêneo. Instale o pytest e execute a partir da raiz:

```bash
python -m pip install pytest
python -m pytest tests/
```

Esse teste não substitui uma validação numérica mais extensa nem uma comparação com medições de campo.

## Arquivos já incluídos

O repositório entregue contém bases CSV, modelos `.joblib` previamente treinados e resultados/gráficos de execuções anteriores. É possível experimentar predições com esses modelos sem refazer todo o treinamento, desde que as dependências estejam instaladas e os arquivos permaneçam nos caminhos esperados.

## Git e boas práticas

O ambiente virtual, caches Python, arquivos locais de IDE e `.env` são ignorados. Para iniciar o versionamento:

```bash
git init
git add .
git commit -m "docs: organiza projeto e documentacao"
```

Se `.venv/` já tiver sido adicionada ao Git anteriormente, o `.gitignore` **não a remove automaticamente** do índice. Nesse caso:

```bash
git rm -r --cached .venv
git add .gitignore
git commit -m "chore: remove ambiente virtual do versionamento"
```

> **Observação:** os datasets, modelos treinados e resultados foram preservados e não são ignorados por padrão. Antes de publicar, confira o tamanho desses arquivos e considere Git LFS para arquivos grandes.

## Limitações e próximos passos

- Ampliar a validação do operador direto, inclusive com estudos de convergência numérica.
- Avaliar o desempenho em diferentes configurações geoelétricas e níveis de ruído.
- Investigar generalização para dados de campo e limitações de não unicidade da inversão.
- Registrar versões e sementes para aumentar a reprodutibilidade experimental.

## Licença

Consulte o arquivo [`LICENSE`](LICENSE) incluído no repositório.
