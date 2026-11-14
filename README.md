# Predição de Demanda de Bicicletas Compartilhadas com Rede Bayesiana

Rede bayesiana construída manualmente para estimar a demanda horária de bicicletas compartilhadas em Seul (Baixa, Moderada ou Alta) a partir de variáveis de clima e de calendário.

- **Dataset:** Seoul Bike Sharing Demand, do [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand). Uma cópia está em `dataset/SeoulBikeData.csv`.
- **Modalidades de análise:** preditiva, diagnóstica e descritiva.
- **Inferência:** exata, por Eliminação de Variáveis (`pgmpy`).

A estrutura e as CPTs foram definidas manualmente no `modelo.json`, sem aprendizagem automática. O notebook `notebooks/01_analise_exploratoria.ipynb` usa o dataset para fundamentar a discretização e as probabilidades.

## Organização

```
modelo.json                 Especificação completa da rede (variáveis, estrutura, CPTs, cenários)
src/rede.py                 Constrói a rede no pgmpy a partir do modelo.json e gera o gráfico do DAG
src/inferencia.py           MotorInferencia: consultas validadas, distribuição conjunta, comparação de evidências
src/cenarios.py             Executa os cenários do modelo.json e verifica suas condições de teste
tests/                      Testes automatizados (pytest), um arquivo por módulo de src/
estrutura_rede_bayesiana.png Representação gráfica da rede
notebooks/                  Análise exploratória do dataset
```

## Instalação

Requer Python 3.10 ou superior.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Execução

Todos os comandos são executados na raiz do projeto.

```bash
python src/rede.py         # resumo estrutural da rede e geração de estrutura_rede_bayesiana.png
python src/inferencia.py   # exemplo: priori da demanda e comparação clima ideal x chuva
python src/cenarios.py     # executa os 4 cenários, compara com a base e verifica as condições
python -m pytest -v        # executa todos os testes automatizados
```

`src/cenarios.py` termina com código de saída diferente de zero se alguma condição de teste de um cenário não for atendida.

Consulta com evidências próprias:

```python
from src.inferencia import MotorInferencia

motor = MotorInferencia()
motor.consultar(["demanda"], {"estacao": "Verao", "horario": "Pico", "precipitacao": "Chuva"})
```

## Testes automatizados

Os testes usam pytest, sem arquivo de configuração. Rode `python -m pytest` na raiz do projeto: o `-m` coloca a raiz no caminho de importação, o que permite o `from src...` dos testes (o comando `pytest` sozinho não faz isso). Os testes só leem os arquivos entregues; os de modelos inválidos usam cópias temporárias do `modelo.json`.

Os valores esperados não são tirados da própria implementação. Eles são conferidos de três formas:

1. **CPTs do `modelo.json`:** quando todos os pais de uma variável estão observados, a posterior tem de ser a linha correspondente da CPT.
2. **Cálculo manual:** os resultados esperados dos cenários 1 (marginalização) e 3 (linha da CPT) são recalculados no teste a partir das CPTs.
3. **Enumeração independente** (função `enumerar` em `test_cenarios.py`): soma a distribuição conjunta P(x1, ..., xn) = Π P(xi | pais(xi)) sobre todas as combinações compatíveis com as evidências, sem usar o pgmpy.

| Arquivo | O que verifica |
|---|---|
| `test_rede.py` | **Estrutura:** ≥ 10 variáveis com nome, descrição e estados únicos e interpretados; nós e arestas da rede iguais ao JSON; ausência de ciclos; geração do gráfico. **Probabilidades:** sem chaves duplicadas no JSON; CPTs cobrem todas as configurações dos pais; cada distribuição tem todos os estados, valores em [0, 1] e soma 1. **Modelos inválidos rejeitados:** ciclo, pai inexistente, campo obrigatório ausente, configuração de pais ausente, estado ausente, soma ≠ 1, probabilidade negativa. |
| `test_inferencia.py` | Distribuições normalizadas; pais observados reproduzem a linha da CPT (108 configurações); regra de Bayes; d-separação; *explaining away*; conjunta consistente com as marginais; resultados determinísticos e independentes do diretório atual. **Consultas inválidas rejeitadas:** variável ou estado inexistente, estado com caixa errada, variável consultada e observada ao mesmo tempo, consulta vazia ou duplicada, evidências fora de dicionário. |
| `test_cenarios.py` | ≥ 4 cenários distintos com todos os campos; cada cenário atende às condições de `resultados_esperados`; a limitação documentada em `expectativa_nao_confirmada` continua divergindo; resultados iguais aos da enumeração independente; esperados dos cenários 1 e 3 derivados das CPTs; `src/cenarios.py` executa sem falhas pela linha de comando; o verificador aprova condições verdadeiras, reprova as falsas e rejeita condições desconhecidas. |

Exemplos de testes e resultados esperados:

| Teste | Esperado | Fundamento |
|---|---|---|
| `P(demanda \| Favoravel, Pico, Sim, Sim)` | Baixa 0.0159, Moderada 0.5873, Alta 0.3968 | Pais todos observados: é a linha `Favoravel\|Pico\|Sim\|Sim` da CPT |
| `P(demanda = Alta \| cenário 1)` | 0.6771 | 0.02·0.0418 + 0.10·0.5671 + 0.88·0.7040 |
| `P(precipitacao \| Inverno, Pico, vento = Forte)` | igual a `P(precipitacao \| Inverno, Pico)` | vento e precipitação são d-separados dados estação e horário |
| `P(feriado = Sim \| demanda = Baixa, conforto = Desfavoravel)` | menor que `P(feriado = Sim \| demanda = Baixa)` | *explaining away*: o clima já explica a demanda baixa |
| `consultar(["demanda"], {"precipitacao": "Granizo"})` | `ValueError` | estado fora dos definidos para a variável |

## Formato do `modelo.json`

| Campo | Conteúdo |
|---|---|
| `identificacao_projeto` | título, tema, dataset, fonte, problema e modalidades de análise |
| `metadados_execucao` | linguagem, versões mínimas das bibliotecas (`bibliotecas`), versões com que os testes foram validados (`versoes_validadas`) e algoritmo de inferência |
| `variaveis` | lista com `id`, `nome`, `descricao`, `estados` (ordem usada em toda a rede) e `interpretacao_estados` |
| `estrutura` | para cada variável: `pais` (lista ordenada) e `justificativa` da dependência |
| `probabilidades` | CPT de cada variável, indexada pelo `id` |
| `cenarios` | cenários de utilização (ver abaixo) |

**CPTs.** Variáveis raiz têm `distribuicao: {estado: p}`. As demais têm `condicionais`, cuja chave é a configuração dos pais unida por `|`, na mesma ordem de `pais`. Por exemplo, em `temperatura` (pais `["estacao", "radiacao_solar"]`), a chave `"Inverno|Alta"` contém `P(temperatura | estacao = Inverno, radiacao_solar = Alta)`. Todas as configurações e todos os estados aparecem explicitamente.

**Cenários.** Cada cenário tem `id`, `titulo`, `modalidade`, `pergunta`, `evidencias` (`{variavel: estado}`), `evidencias_base` (obrigatória: situação de referência, sem a informação que o cenário quer avaliar), `variaveis_consultadas`, `resultados_esperados`, `fundamentacao_resultados_esperados`, `interpretacao` e `justificativa_relevancia`. Opcionalmente, `expectativa_nao_confirmada` registra uma expectativa intuitiva que a rede não confirma (limitação do modelo): usa as mesmas condições abaixo, mas precisa *falhar*. O cenário 4 a usa: esperava-se Demanda Baixa e a rede dá Moderada.

`resultados_esperados` mapeia cada variável consultada para condições de teste:

| Condição | Significado |
|---|---|
| `"<estado>": p` | `P(estado \| evidencias)` igual a `p` (tolerância 0.001) |
| `"estado_mais_provavel": "<estado>"` | estado de maior probabilidade posterior |
| `"probabilidade_minima_esperada": p` | `P(estado_mais_provavel \| evidencias) ≥ p` |
| `"aumenta_em_relacao_base": [estados]` | `P(estado \| evidencias) > P(estado \| evidencias_base)` |
| `"diminui_em_relacao_base": [estados]` | `P(estado \| evidencias) < P(estado \| evidencias_base)` |

## Bibliotecas

| Biblioteca | Contribuição |
|---|---|
| `pgmpy` | representação da rede (`DiscreteBayesianNetwork`, `TabularCPD`), validação do modelo (`check_model`) e inferência exata (`VariableElimination`) |
| `networkx` | verificação de aciclicidade e desenho do grafo |
| `matplotlib` | geração da figura `estrutura_rede_bayesiana.png` |
| `numpy` | normalização das distribuições nas consultas |
| `pytest` | execução dos testes automatizados |
| `pandas`, `seaborn` | análise exploratória no notebook (não são usados pela rede) |
