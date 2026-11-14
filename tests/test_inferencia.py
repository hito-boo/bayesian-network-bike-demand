"""
Testes de src/inferencia.py: as consultas produzem distribuições normalizadas,
respeitam propriedades probabilísticas verificáveis, são reprodutíveis e
rejeitam evidências inválidas.

Execução (na raiz do projeto): python -m pytest -v
"""
from __future__ import annotations

import pytest

from src.inferencia import MotorInferencia

MOTOR = MotorInferencia()


def probabilidade(var, estado, evidencias=None):
    """Retorna P(var = estado | evidencias) com 10 casas decimais."""
    return MOTOR.consultar([var], evidencias, casas_decimais=10)[var][estado]


def test_distribuicoes_sao_normalizadas():
    evidencias = {"estacao": "Inverno", "demanda": "Baixa"}
    variaveis = [v for v in MOTOR.mapa_estados if v not in evidencias]

    for var, dist in MOTOR.consultar(variaveis, evidencias).items():
        assert list(dist) == MOTOR.mapa_estados[var]
        assert all(0.0 <= p <= 1.0 for p in dist.values())
        assert sum(dist.values()) == pytest.approx(1.0, abs=1e-3)


def test_pais_observados_reproduzem_a_linha_da_cpt():
    # Por definição, P(X | pais(X) = c) é a linha 'c' da CPT de X no modelo.json
    for var, bloco in MOTOR.especificacao["probabilidades"].items():
        for config, esperado in bloco.get("condicionais", {}).items():
            obtido = MOTOR.consultar([var], dict(zip(bloco["pais"], config.split("|"))))[var]
            assert obtido == pytest.approx(esperado, abs=1e-4), f"{var} | {config}"


def test_regra_de_bayes():
    # P(demanda | conforto) * P(conforto) = P(conforto | demanda) * P(demanda)
    p_conforto = probabilidade("conforto_climatico", "Desfavoravel")
    p_demanda = probabilidade("demanda", "Baixa")
    p_demanda_dado_conforto = probabilidade("demanda", "Baixa", {"conforto_climatico": "Desfavoravel"})
    p_conforto_dado_demanda = probabilidade("conforto_climatico", "Desfavoravel", {"demanda": "Baixa"})

    assert p_demanda_dado_conforto * p_conforto == pytest.approx(p_conforto_dado_demanda * p_demanda)


def test_vento_nao_altera_precipitacao_dados_estacao_e_horario():
    # vento e precipitacao são d-separados quando seus ancestrais comuns são observados
    base = {"estacao": "Inverno", "horario": "Pico"}
    com_vento = {**base, "vento": "Forte"}
    assert probabilidade("precipitacao", "Neve", com_vento) == pytest.approx(probabilidade("precipitacao", "Neve", base))


def test_clima_desfavoravel_explica_a_demanda_baixa():
    # Explaining away (feriado -> demanda <- conforto_climatico): sabendo que o clima
    # estava ruim, a demanda baixa fica explicada e a crença em feriado diminui
    so_demanda = probabilidade("feriado", "Sim", {"demanda": "Baixa"})
    com_clima = probabilidade("feriado", "Sim", {"demanda": "Baixa", "conforto_climatico": "Desfavoravel"})
    assert com_clima < so_demanda


def test_distribuicao_conjunta_e_consistente_com_as_marginais():
    evidencias = {"estacao": "Inverno", "horario": "Pico"}
    conjunta = MOTOR.consultar_conjunta(["precipitacao", "conforto_climatico"], evidencias, casas_decimais=10)
    marginal = MOTOR.consultar(["precipitacao"], evidencias, casas_decimais=10)["precipitacao"]

    assert sum(conjunta.values()) == pytest.approx(1.0)
    for estado, p in marginal.items():
        # Somar a conjunta sobre os estados de conforto resulta na marginal de precipitação
        soma = sum(valor for chave, valor in conjunta.items() if chave.startswith(f"{estado}|"))
        assert soma == pytest.approx(p)


def test_resultados_sao_deterministicos_e_independem_do_diretorio_atual(tmp_path, monkeypatch):
    # Reprodutibilidade: uma nova instância, criada em outro diretório, gera os mesmos resultados
    monkeypatch.chdir(tmp_path)
    evidencias = {"estacao": "Verao", "horario": "Comercial", "feriado": "Sim"}
    assert MotorInferencia().consultar(["demanda"], evidencias) == MOTOR.consultar(["demanda"], evidencias)


CONSULTAS_INVALIDAS = {
    "variavel_inexistente": (["chuva_forte"], None),
    "evidencia_inexistente": (["demanda"], {"neblina": "Sim"}),
    "estado_inexistente": (["demanda"], {"precipitacao": "Granizo"}),
    "estado_com_caixa_errada": (["demanda"], {"feriado": "sim"}),
    "variavel_consultada_e_observada": (["demanda"], {"demanda": "Alta"}),
    "consulta_vazia": ([], None),
    "consulta_duplicada": (["demanda", "demanda"], None),
    "evidencias_fora_de_dicionario": (["demanda"], [("feriado", "Sim")]),
}


@pytest.mark.parametrize("variaveis, evidencias", list(CONSULTAS_INVALIDAS.values()), ids=list(CONSULTAS_INVALIDAS))
def test_consulta_invalida_e_rejeitada(variaveis, evidencias):
    with pytest.raises(ValueError):
        MOTOR.consultar(variaveis, evidencias)
