"""
Testes de src/cenarios.py: execução dos cenários de utilização do modelo.json,
conferência dos resultados com uma inferência por enumeração independente do
pgmpy e execução do script de cenários pela linha de comando.

Execução (na raiz do projeto): python -m pytest -v
"""
from __future__ import annotations

import itertools
import math
import os
import subprocess
import sys

import pytest

from src.cenarios import executar_cenario, verificar_condicoes
from src.inferencia import MotorInferencia
from src.rede import CAMINHO_PADRAO_JSON

MOTOR = MotorInferencia()
CENARIOS = MOTOR.especificacao["cenarios"]
IDS_CENARIOS = [c["id"] for c in CENARIOS]
ESTADOS = MOTOR.mapa_estados
PAIS = {item["variavel"]: item["pais"] for item in MOTOR.especificacao["estrutura"]}
CPTS = MOTOR.especificacao["probabilidades"]

CAMPOS_OBRIGATORIOS = [
    "id",
    "titulo",
    "pergunta",
    "evidencias",
    "evidencias_base",
    "variaveis_consultadas",
    "resultados_esperados",
    "interpretacao",
    "justificativa_relevancia",
]


def prob_local(var, valores):
    """Retorna P(var | pais(var)) para a atribuição 'valores', lida diretamente da CPT do modelo.json."""
    if not PAIS[var]:
        return CPTS[var]["distribuicao"][valores[var]]
    return CPTS[var]["condicionais"]["|".join(valores[p] for p in PAIS[var])][valores[var]]


def enumerar(var, evidencias):
    """
    Inferência por enumeração exaustiva, sem pgmpy: soma a distribuição conjunta
    P(x1, ..., xn) = produto de P(xi | pais(xi)) sobre todas as combinações
    compatíveis com as evidências e normaliza o resultado.
    """
    livres = [v for v in ESTADOS if v not in evidencias]
    acumulado = dict.fromkeys(ESTADOS[var], 0.0)

    for combinacao in itertools.product(*(ESTADOS[v] for v in livres)):
        valores = {**evidencias, **dict(zip(livres, combinacao))}
        acumulado[valores[var]] += math.prod(prob_local(v, valores) for v in ESTADOS)

    total = sum(acumulado.values())
    return {estado: p / total for estado, p in acumulado.items()}


# --- Execução dos cenários ---

def test_ha_ao_menos_quatro_cenarios_distintos():
    assert len(CENARIOS) >= 4
    assert len(set(IDS_CENARIOS)) == len(IDS_CENARIOS)


@pytest.mark.parametrize("cenario", CENARIOS, ids=IDS_CENARIOS)
def test_cenario_possui_campos_obrigatorios(cenario):
    for campo in CAMPOS_OBRIGATORIOS:
        assert cenario.get(campo), f"Campo '{campo}' ausente ou vazio."


@pytest.mark.parametrize("cenario", CENARIOS, ids=IDS_CENARIOS)
def test_cenario_atende_as_condicoes_de_teste(cenario):
    assert executar_cenario(MOTOR, cenario)["falhas"] == []


def test_expectativa_nao_confirmada_de_fato_diverge():
    # A limitação documentada no cenário 4 precisa continuar real: a rede não prevê Demanda Baixa
    com_limitacao = [c for c in CENARIOS if "expectativa_nao_confirmada" in c]
    assert com_limitacao
    for cenario in com_limitacao:
        assert executar_cenario(MOTOR, cenario)["divergencias"] != [], cenario["id"]


@pytest.mark.parametrize("cenario", CENARIOS, ids=IDS_CENARIOS)
def test_cenario_confere_com_a_enumeracao_independente(cenario):
    for var, dist in executar_cenario(MOTOR, cenario)["resultados"].items():
        assert dist == pytest.approx(enumerar(var, cenario["evidencias"]), abs=1e-4), var


def test_resultado_esperado_do_cenario_1_deriva_das_cpts():
    # Conforto tem todos os pais observados (linha da CPT); a demanda marginaliza o conforto
    esperados = CENARIOS[0]["resultados_esperados"]
    conforto = CPTS["conforto_climatico"]["condicionais"]["Amena|Nenhuma|Fraco"]

    for estado, p in conforto.items():
        assert esperados["conforto_climatico"][estado] == p
    for estado in ESTADOS["demanda"]:
        p = sum(conforto[c] * CPTS["demanda"]["condicionais"][f"{c}|Pico|Nao|Sim"][estado] for c in conforto)
        assert esperados["demanda"][estado] == pytest.approx(p, abs=1e-4)


def test_resultado_esperado_do_cenario_3_e_a_linha_da_cpt():
    # Todos os pais de demanda estão observados no cenário 3
    esperados = CENARIOS[2]["resultados_esperados"]["demanda"]
    for estado, p in CPTS["demanda"]["condicionais"]["Favoravel|Pico|Sim|Sim"].items():
        assert esperados[estado] == p


def test_script_de_cenarios_executa_sem_falhas():
    # Reprodutibilidade: os cenários rodam pela linha de comando só com os arquivos entregues
    processo = subprocess.run(
        [sys.executable, os.path.join("src", "cenarios.py")],
        cwd=CAMINHO_PADRAO_JSON.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert processo.returncode == 0, processo.stderr


# --- O verificador de condições não pode ser vacuamente verdadeiro ---

RESULTADO = {"demanda": {"Baixa": 0.2, "Moderada": 0.3, "Alta": 0.5}}
BASE = {"demanda": {"Baixa": 0.4, "Moderada": 0.3, "Alta": 0.3}}

CONDICOES_FALSAS = {
    "valor_exato_errado": {"Alta": 0.6},
    "estado_mais_provavel_errado": {"estado_mais_provavel": "Baixa"},
    "abaixo_do_minimo": {"estado_mais_provavel": "Alta", "probabilidade_minima_esperada": 0.7},
    "deveria_aumentar": {"aumenta_em_relacao_base": ["Baixa"]},
    "deveria_diminuir": {"diminui_em_relacao_base": ["Alta"]},
    "sem_variacao": {"aumenta_em_relacao_base": ["Moderada"]},
}


@pytest.mark.parametrize("condicao", list(CONDICOES_FALSAS.values()), ids=list(CONDICOES_FALSAS))
def test_verificador_reprova_condicao_falsa(condicao):
    assert verificar_condicoes(RESULTADO, BASE, {"demanda": condicao}) != []


def test_verificador_aprova_condicoes_verdadeiras():
    condicoes = {
        "Baixa": 0.2,
        "estado_mais_provavel": "Alta",
        "probabilidade_minima_esperada": 0.5,
        "aumenta_em_relacao_base": ["Alta"],
        "diminui_em_relacao_base": ["Baixa"],
    }
    assert verificar_condicoes(RESULTADO, BASE, {"demanda": condicoes}) == []


def test_verificador_rejeita_condicao_desconhecida():
    with pytest.raises(ValueError):
        verificar_condicoes(RESULTADO, BASE, {"demanda": {"quase_certo": True}})
