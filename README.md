# TCC — Inversão Geoelétrica 1D com Redes Neurais

**Versão principal:** V8, sem TRF e sem Levenberg–Marquardt na previsão. Apenas três arquivos de pesos são usados: `neural_v5.pt` (MLP de base), `neural_v7_residual.pt` (correção neural) e `neural_v8_physics_correction.pt` (correção neural guiada pelo resíduo físico). V5 e V7 são **componentes internos** da V8, não métodos concorrentes apresentados no artigo.

## Instalar e executar

```bash
python -m pip install -r requirements.txt
python -m src.predict --rho1 100 --rho2 30 --rho3 250 --h1 5 --h2 12 --outdir results/minha_previsao
python -m src.predict --rho1 100 --rho2 30 --rho3 250 --h1 5 --h2 12 --noise-pct 2 --outdir results/minha_previsao_ruidosa
python -m src.predict --csv results/exemplos/entrada_sem_ruido.csv --outdir results/meu_csv
python -m src.examples
python -m src.evaluate
python -m pytest -q
# Opcional: novo treinamento da correção V8 usando checkpoints V5/V7 congelados
python -m src.train_v8 --epochs 10 --n-scenario 2500 --skip-eval
```

Cada previsão cria `curva_v8.png`, `curvas.csv` e `metricas.json`. O script de exemplos já produziu gráficos para **quatro cenários, cada um sem ruído e com 2% de ruído**, disponíveis em `results/exemplos`.

## CSV

As colunas devem chamar-se **`AB2_m,rho_apparent_ohmm`**, conter **exatamente 20 linhas** com espaçamentos `np.geomspace(1,200,20)` em metros, ordenadas, e valores positivos em Ω·m. Isso é exigido pelos checkpoints treinados. Medições com outra configuração de eletrodos **não devem ser reinterpretadas como equivalentes**; exigem tratamento/validação específicos.

## Física da simulação

- SEV, arranjo Schlumberger aproximado para MN/AB pequeno; três camadas horizontais; 20 aberturas `AB/2` entre 1 e 200 m, ou `AB` total entre 2 e 400 m.
- `field_protocol.py` inclui as convenções SI, espaçamentos MN nominais e o fator geométrico, mas o operador direto é a aproximação Schlumberger baseada em AB/2; **não simula explicitamente o potencial para MN finito**.
- `realistic_data.py` define cenários hipotéticos (não são medições de campo). Ruído lognormal multiplicativo de 2% é uma perturbação sintética, não um retrato completo do erro instrumental.
- A abertura de 400 m não define a profundidade efetiva; h1 e h2 são espessuras e a terceira camada é semiespaço. ASTM D6431-25 fornece diretrizes gerais sobre eletrorresistividade DC, não exige 20 pontos ou 400 m.

## Treinamento e reprodutibilidade

Os checkpoints V5/V7/V8 são fornecidos **já treinados**. Este pacote reúne apenas os componentes usados para executar e avaliar a V8; os scripts/artefatos de V1–V7 e treinamento integral desde o primeiro modelo foram excluídos da distribuição limpa. Para reproduzir o **treinamento completo** desde o início, é necessário o arquivo histórico do projeto. O ajuste da correção V8 pode ser reexecutado a partir desses checkpoints com `python -m src.train_v8`, salvando por padrão um modelo candidato sem sobrescrever o modelo validado. A avaliação reservada da V8 pode ser repetida sem retreinamento usando `python -m src.evaluate`.

## Resultados já verificados (holdout final)

`results/metricas/neural_v8_locked_holdout.json`: 400 modelos novos por cenário (seed 99103), sem ruído e com 2% de ruído; no último caso, o desajuste compara o modelo reconstruído com a **curva limpa**, não com a entrada perturbada. Para a V8 com duas passagens: medianas sem ruído 1,478%, 1,310%, 1,314%, 1,329%; com 2% ruído 1,809%, 1,730%, 1,769%, 1,761% (original, úmido, seco/rochoso, condutivo). Meta inferior a 1% **não foi atingida na mediana**. O caso ilustrativo [100,30,250,5,12] obteve ~0,606% sem ruído, não é representativo da média.

## Organização

- `src/`: operador direto, modelos neurais, simulação, predição, exemplos e avaliação.
- `models/`: três checkpoints estritamente necessários.
- `results/exemplos/`: curvas e métricas de casos reprodutíveis.
- `results/metricas/`: avaliação reservada e histórico do treinamento da V8.
- `artigo/`: materiais LaTeX finais (também entregues em ZIP independente).
- `tests/`: verificações automatizadas.
