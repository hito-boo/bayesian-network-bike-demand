"""
Testes de src/rede.py: validação estrutural da rede, validação das
probabilidades (CPTs) do modelo.json e rejeição de modelos inválidos.

Execução (na raiz do projeto): python -m pytest -v
"""
from __future__ import annotations

import itertools
import json

import matplotlib
import networkx as nx
import pytest

from src.rede import CAMINHO_PADRAO_JSON, carregar_especificacao_json, construir_rede, plotar_rede

MODELO, DADOS = construir_rede()
ESTADOS = {v["id"]: v["estados"] for v in DADOS["variaveis"]}
PAIS = {item["variavel"]: item["pais"] for item in DADOS["estrutura"]}


def pais_de(dados, var):
    """Retorna a lista de pais de 'var' na seção 'estrutura' do modelo."""
    return next(item["pais"] for item in dados["estrutura"] if item["variavel"] == var)


def cpt(dados, var):
    """Retorna o bloco de probabilidades de 'var' na seção 'probabilidades' do modelo."""
    return dados["probabilidades"][var]


def construir_com_alteracao(tmp_path, alterar):
    """
    Aplica a função 'alterar' em uma cópia do modelo.json e constrói a rede a
    partir dela, para verificar a rejeição de modelos inválidos.
    """
    dados = carregar_especificacao_json()
    alterar(dados)
    caminho = tmp_path / "modelo.json"
    caminho.write_text(json.dumps(dados), encoding="utf-8")
    return construir_rede(caminho)


# --- Validação estrutural ---

def test_possui_ao_menos_10_variaveis_distintas():
    ids = [v["id"] for v in DADOS["variaveis"]]
    assert len(ids) >= 10
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("var", DADOS["variaveis"], ids=list(ESTADOS))
def test_variavel_possui_estados_definidos_e_interpretados(var):
    assert var["nome"] and var["descricao"]
    assert len(var["estados"]) >= 2
    assert len(set(var["estados"])) == len(var["estados"])
    assert set(var["interpretacao_estados"]) == set(var["estados"])


def test_rede_possui_as_variaveis_e_dependencias_do_json():
    arestas = {(pai, filho) for filho, pais in PAIS.items() for pai in pais}
    assert set(MODELO.nodes()) == set(ESTADOS)
    assert set(MODELO.edges()) == arestas


def test_grafo_e_aciclico():
    assert nx.is_directed_acyclic_graph(MODELO)


def test_representacao_grafica_e_gerada(tmp_path):
    matplotlib.use("Agg")  # salva a figura sem abrir janela
    caminho = plotar_rede(caminho_saida_png=tmp_path / "rede.png")
    assert caminho.stat().st_size > 0


# --- Validação das probabilidades ---

def test_json_nao_possui_chaves_duplicadas():
    def rejeitar_duplicadas(pares):
        chaves = [chave for chave, _ in pares]
        assert len(chaves) == len(set(chaves)), f"Chaves duplicadas: {chaves}"
        return dict(pares)

    json.loads(CAMINHO_PADRAO_JSON.read_text(encoding="utf-8"), object_pairs_hook=rejeitar_duplicadas)


def test_cpts_cobrem_todas_as_configuracoes_dos_pais():
    for var, pais in PAIS.items():
        if pais:
            combinacoes = {"|".join(c) for c in itertools.product(*(ESTADOS[p] for p in pais))}
            assert set(cpt(DADOS, var)["condicionais"]) == combinacoes, var


def test_distribuicoes_tem_todos_os_estados_valores_validos_e_somam_1():
    for var, bloco in DADOS["probabilidades"].items():
        linhas = bloco["condicionais"] if bloco["pais"] else {"a priori": bloco["distribuicao"]}
        for config, dist in linhas.items():
            assert list(dist) == ESTADOS[var], f"{var} | {config}"
            assert all(0.0 <= p <= 1.0 for p in dist.values()), f"{var} | {config}"
            assert sum(dist.values()) == pytest.approx(1.0, abs=1e-4), f"{var} | {config}"


# --- Modelos inválidos devem ser rejeitados ---

# Cada caso indica a exceção esperada e a alteração aplicada em uma cópia do modelo.json
MODELOS_INVALIDOS = {
    "ciclo": (ValueError, lambda d: pais_de(d, "estacao").append("demanda")),
    "pai_inexistente": (KeyError, lambda d: pais_de(d, "vento").append("neblina")),
    "campo_obrigatorio_ausente": (ValueError, lambda d: d.pop("cenarios")),
    "configuracao_de_pais_ausente": (KeyError, lambda d: cpt(d, "vento")["condicionais"].pop("Inverno|Pico")),
    "estado_ausente": (KeyError, lambda d: cpt(d, "precipitacao")["condicionais"]["Alta|Baixa"].pop("Neve")),
    "soma_diferente_de_1": (ValueError, lambda d: cpt(d, "feriado").update(distribuicao={"Nao": 0.7, "Sim": 0.2})),
    "probabilidade_negativa": (ValueError, lambda d: cpt(d, "feriado").update(distribuicao={"Nao": 1.2, "Sim": -0.2})),
}


@pytest.mark.parametrize("erro, alterar", list(MODELOS_INVALIDOS.values()), ids=list(MODELOS_INVALIDOS))
def test_modelo_invalido_e_rejeitado(tmp_path, erro, alterar):
    with pytest.raises(erro):
        construir_com_alteracao(tmp_path, alterar)
