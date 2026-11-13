from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Compatibilidade entre versões recentes e anteriores do pgmpy
try:
    from pgmpy.models import DiscreteBayesianNetwork as ModeloBayesiano
except ImportError:
    from pgmpy.models import BayesianNetwork as ModeloBayesiano

from pgmpy.factors.discrete import TabularCPD


CAMINHO_PADRAO_JSON = Path(__file__).resolve().parent.parent / "modelo.json"
CAMINHO_PADRAO_PNG = Path(__file__).resolve().parent.parent / "estrutura_rede_bayesiana.png"


def carregar_especificacao_json(caminho_json: str | Path = CAMINHO_PADRAO_JSON) -> Dict[str, Any]:
    """
    Lê e valida a existência do arquivo modelo.json que descreve a Rede Bayesiana.
    """
    caminho = Path(caminho_json)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de especificação não encontrado em: {caminho}")

    with open(caminho, mode="r", encoding="utf-8") as f:
        dados = json.load(f)

    campos_obrigatorios = [
        "identificacao_projeto",
        "metadados_execucao",
        "variaveis",
        "estrutura",
        "probabilidades",
        "cenarios",
    ]
    for campo in campos_obrigatorios:
        if campo not in dados:
            raise ValueError(f"Campo obrigatório '{campo}' ausente no arquivo {caminho.name}.")

    return dados


def _montar_matriz_cpd(
    var_id: str,
    estados_var: List[str],
    pais: List[str],
    mapa_estados: Dict[str, List[str]],
    bloco_prob: Dict[str, Any],
) -> List[List[float]]:
    """
    Converte o dicionário legível de probabilidades do modelo.json na matriz 2D
    exigida pelo TabularCPD do pgmpy, respeitando a ordem exata do produto
    cartesiano dos estados dos pais.
    """
    if not pais:
        if "distribuicao" not in bloco_prob:
            raise ValueError(f"Variável raiz '{var_id}' não possui chave 'distribuicao'.")
        dist = bloco_prob["distribuicao"]
        return [[float(dist[estado])] for estado in estados_var]

    if "condicionais" not in bloco_prob:
        raise ValueError(f"Variável '{var_id}' com pais {pais} não possui chave 'condicionais'.")

    condicionais = bloco_prob["condicionais"]
    listas_estados_pais = [mapa_estados[p] for p in pais]
    combinacoes_pais = list(itertools.product(*listas_estados_pais))

    matriz: List[List[float]] = [[] for _ in estados_var]

    for comb in combinacoes_pais:
        chave_comb = "|".join(comb)
        if chave_comb not in condicionais:
            raise KeyError(
                f"Configuração de pais '{chave_comb}' ausente na CPT da variável '{var_id}'."
            )
        dist_condicional = condicionais[chave_comb]
        for idx_estado, estado in enumerate(estados_var):
            if estado not in dist_condicional:
                raise KeyError(
                    f"Estado '{estado}' ausente na configuração '{chave_comb}' da variável '{var_id}'."
                )
            matriz[idx_estado].append(float(dist_condicional[estado]))

    return matriz


def construir_rede(caminho_json: str | Path = CAMINHO_PADRAO_JSON,) -> Tuple[ModeloBayesiano, Dict[str, Any]]:
    """
    Constrói manualmente a Rede Bayesiana no pgmpy a partir das definições
    explícitas do arquivo modelo.json e verifica a consistência do modelo.

    Retorna:
        Tuple[ModeloBayesiano, Dict[str, Any]]: Objeto da rede instanciado e dicionário do JSON.
    """
    dados = carregar_especificacao_json(caminho_json)

    mapa_estados: Dict[str, List[str]] = {
        v["id"]: list(v["estados"]) for v in dados["variaveis"]
    }

    # Extrair arestas direcionadas (Pai -> Filho)
    arestas: List[Tuple[str, str]] = []
    mapa_pais: Dict[str, List[str]] = {}

    for item in dados["estrutura"]:
        filho = item["variavel"]
        pais = list(item.get("pais", []))
        mapa_pais[filho] = pais
        for pai in pais:
            arestas.append((pai, filho))

    # Instanciar o grafo da Rede Bayesiana
    modelo = ModeloBayesiano()
    modelo.add_nodes_from(mapa_estados.keys())
    modelo.add_edges_from(arestas)

    # Verificar ausência de ciclos antes de adicionar CPDs
    if not nx.is_directed_acyclic_graph(modelo):
        raise ValueError("A estrutura definida contém ciclos e não forma um DAG válido.")

    # Instanciar e anexar todas as Tabelas de Probabilidade Condicional (CPTs)
    probabilidades_json = dados["probabilidades"]

    for var_id, estados_var in mapa_estados.items():
        if var_id not in probabilidades_json:
            raise KeyError(f"Variável '{var_id}' não possui CPT definida em 'probabilidades'.")

        pais = mapa_pais.get(var_id, [])
        bloco_prob = probabilidades_json[var_id]

        valores_matriz = _montar_matriz_cpd(
            var_id=var_id,
            estados_var=estados_var,
            pais=pais,
            mapa_estados=mapa_estados,
            bloco_prob=bloco_prob,
        )

        nomes_estados_cpd = {var_id: estados_var}
        for p in pais:
            nomes_estados_cpd[p] = mapa_estados[p]

        if not pais:
            cpd = TabularCPD(
                variable=var_id,
                variable_card=len(estados_var),
                values=valores_matriz,
                state_names=nomes_estados_cpd,
            )
        else:
            cpd = TabularCPD(
                variable=var_id,
                variable_card=len(estados_var),
                values=valores_matriz,
                evidence=pais,
                evidence_card=[len(mapa_estados[p]) for p in pais],
                state_names=nomes_estados_cpd,
            )

        modelo.add_cpds(cpd)

    # Validar consistência global no pgmpy (somas = 1, dimensões e estados)
    modelo.check_model()

    return modelo, dados


def obter_resumo_rede(modelo: ModeloBayesiano, dados: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retorna métricas estruturais e probabilísticas da rede para inspeção e testes.
    """
    total_parametros_cpt = 0
    for cpd in modelo.get_cpds():
        total_parametros_cpt += int(np.prod(cpd.values.shape))

    return {
        "titulo_projeto": dados["identificacao_projeto"]["titulo"],
        "num_variaveis": len(modelo.nodes()),
        "num_arestas": len(modelo.edges()),
        "eh_dag": nx.is_directed_acyclic_graph(modelo),
        "modelo_valido_pgmpy": bool(modelo.check_model()),
        "nos_raizes": [n for n, d in modelo.in_degree() if d == 0],
        "nos_folhas": [n for n, d in modelo.out_degree() if d == 0],
        "total_entradas_cpt": total_parametros_cpt,
    }


def plotar_rede(
    caminho_json: str | Path = CAMINHO_PADRAO_JSON,
    caminho_saida_png: str | Path = CAMINHO_PADRAO_PNG,
    exibir: bool = False,
) -> Path:
    """
    Gera uma representação gráfica legível e hierárquica (em camadas causais)
    da estrutura da Rede Bayesiana e salva como imagem PNG (Requisito 5.7).
    """
    modelo, _ = construir_rede(caminho_json)
    caminho_png = Path(caminho_saida_png)

    # Coordenadas fixas organizadas em 5 níveis hierárquicos para máxima legibilidade
    posicoes: Dict[str, Tuple[float, float]] = {
        # Nível 4 (Topo - Raízes Climáticas/Temporais)
        "estacao": (1.5, 4.2),
        "horario": (4.5, 4.2),
        "feriado": (7.2, 2.0),
        "dia_ativo": (9.2, 2.0),
        # Nível 3 (Clima Primário)
        "radiacao_solar": (0.8, 3.0),
        "umidade": (3.0, 3.0),
        "vento": (5.2, 3.0),
        # Nível 2 (Clima Derivado)
        "temperatura": (1.2, 1.8),
        "precipitacao": (3.2, 1.8),
        "visibilidade": (5.2, 1.8),
        # Nível 1 (Variável Intermediária de Domínio)
        "conforto_climatico": (3.2, 0.8),
        # Nível 0 (Variável Alvo)
        "demanda": (5.8, -0.3),
    }

    rotulos = {
        "estacao": "estacao\n(Inverno, Meia, Verao)",
        "horario": "horario\n(Madrug., Comerc., Pico)",
        "feriado": "feriado\n(Nao, Sim)",
        "dia_ativo": "dia_ativo\n(Nao, Sim)",
        "radiacao_solar": "radiacao_solar\n(Baixa, Mod., Alta)",
        "umidade": "umidade\n(Baixa, Mod., Alta)",
        "vento": "vento\n(Fraco, Forte)",
        "temperatura": "temperatura\n(Baixa, Amena, Alta)",
        "precipitacao": "precipitacao\n(Nenhuma, Chuva, Neve)",
        "visibilidade": "visibilidade\n(Baixa, Alta)",
        "conforto_climatico": "conforto_climatico\n(Desfav., Mod., Fav.)",
        "demanda": "demanda\n(Baixa, Moderada, Alta)",
    }

    cores_por_no = {
        "estacao": "#AED6F1",
        "horario": "#AED6F1",
        "feriado": "#AED6F1",
        "dia_ativo": "#AED6F1",
        "radiacao_solar": "#A9DFBF",
        "umidade": "#A9DFBF",
        "vento": "#A9DFBF",
        "temperatura": "#F9E79F",
        "precipitacao": "#F9E79F",
        "visibilidade": "#F9E79F",
        "conforto_climatico": "#F5CBA7",
        "demanda": "#F1948A",
    }

    grafo_nx = nx.DiGraph()
    grafo_nx.add_nodes_from(modelo.nodes())
    grafo_nx.add_edges_from(modelo.edges())

    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    ax.set_title(
        "Estrutura da Rede Bayesiana - Demanda de Bicicletas em Seul (DAG)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    lista_nos = list(grafo_nx.nodes())
    lista_cores = [cores_por_no.get(n, "#D5DBDB") for n in lista_nos]

    nx.draw_networkx_nodes(
        grafo_nx,
        pos=posicoes,
        nodelist=lista_nos,
        node_color=lista_cores,
        node_size=3600,
        edgecolors="#2C3E50",
        linewidths=1.5,
        ax=ax,
    )

    nx.draw_networkx_labels(
        grafo_nx,
        pos=posicoes,
        labels=rotulos,
        font_size=8.5,
        font_weight="bold",
        font_color="#1B2631",
        ax=ax,
    )

    nx.draw_networkx_edges(
        grafo_nx,
        pos=posicoes,
        edge_color="#34495E",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=20,
        width=1.8,
        connectionstyle="arc3,rad=0.06",
        min_source_margin=28,
        min_target_margin=28,
        ax=ax,
    )

    legenda = [
        mpatches.Patch(color="#AED6F1", label="Nós Raízes (Calendário, Relógio e Operação)"),
        mpatches.Patch(color="#A9DFBF", label="Clima Primário (Radiação, Umidade, Vento)"),
        mpatches.Patch(color="#F9E79F", label="Clima Derivado (Temperatura, Precipitação, Visibilidade)"),
        mpatches.Patch(color="#F5CBA7", label="Variável Intermediária (Conforto Climático)"),
        mpatches.Patch(color="#F1948A", label="Variável Alvo (Demanda de Bicicletas)"),
    ]
    ax.legend(handles=legenda, loc="lower left", frameon=True, fontsize=9.5)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(caminho_png, dpi=150, bbox_inches="tight")
    if exibir:
        plt.show()
    plt.close(fig)

    return caminho_png


if __name__ == "__main__":
    rede, spec = construir_rede()
    resumo = obter_resumo_rede(rede, spec)
    caminho_img = plotar_rede()

    print("--- RESUMO ESTRUTURAL DA REDE BAYESIANA ---")
    for k, v in resumo.items():
        print(f"  {k}: {v}")
    print(f"  Gráfico do DAG salvo em: {caminho_img}")
