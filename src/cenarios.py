from __future__ import annotations

from typing import Any, Dict, List

try:
    from src.inferencia import MotorInferencia
except ImportError:
    from inferencia import MotorInferencia


TOLERANCIA = 1e-3


def verificar_condicoes(
    resultados: Dict[str, Dict[str, float]],
    resultados_base: Dict[str, Dict[str, float]],
    esperados: Dict[str, Dict[str, Any]],
) -> List[str]:
    """
    Confronta as distribuições obtidas em um cenário com as condições de teste
    declaradas em 'resultados_esperados' no modelo.json.

    Condições aceitas para cada variável consultada:
        <estado>: p                         -> P(estado | evidencias) igual a p (± TOLERANCIA)
        estado_mais_provavel: estado        -> estado de maior probabilidade posterior
        probabilidade_minima_esperada: p    -> P(estado_mais_provavel | evidencias) >= p
        aumenta_em_relacao_base: [estados]  -> P(estado | evidencias) > P(estado | evidencias_base)
        diminui_em_relacao_base: [estados]  -> P(estado | evidencias) < P(estado | evidencias_base)

    Retorna:
        List[str]: Descrição de cada condição não atendida (lista vazia = cenário aprovado).
    """
    falhas: List[str] = []

    for var, condicoes in esperados.items():
        dist, base = resultados[var], resultados_base[var]

        for condicao, valor in condicoes.items():
            if condicao == "estado_mais_provavel":
                obtido = max(dist, key=dist.get)
                if obtido != valor:
                    falhas.append(f"{var}: estado mais provável '{obtido}', esperado '{valor}'.")

            elif condicao == "probabilidade_minima_esperada":
                estado = condicoes["estado_mais_provavel"]
                if dist[estado] < valor:
                    falhas.append(f"{var}: P({estado}) = {dist[estado]}, abaixo do mínimo {valor}.")

            elif condicao == "aumenta_em_relacao_base":
                falhas += [
                    f"{var}: P({e}) não aumentou em relação à base ({base[e]} -> {dist[e]})."
                    for e in valor
                    if dist[e] <= base[e]
                ]

            elif condicao == "diminui_em_relacao_base":
                falhas += [
                    f"{var}: P({e}) não diminuiu em relação à base ({base[e]} -> {dist[e]})."
                    for e in valor
                    if dist[e] >= base[e]
                ]

            # Demais chaves são estados da variável, comparados com a probabilidade exata esperada
            elif condicao in dist:
                if abs(dist[condicao] - valor) > TOLERANCIA:
                    falhas.append(f"{var}: P({condicao}) = {dist[condicao]}, esperado {valor}.")

            else:
                raise ValueError(f"Condição '{condicao}' desconhecida em 'resultados_esperados' de '{var}'.")

    return falhas


def executar_cenario(motor: MotorInferencia, cenario: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executa um cenário de utilização do modelo.json: consulta a rede com as
    evidências do cenário e com as da situação de referência ('evidencias_base')
    e verifica as condições de teste definidas para ele.

    Retorna:
        Dict[str, Any]: Distribuições com as evidências ('resultados'), distribuições
        da situação de referência ('resultados_base') e condições não atendidas ('falhas').
    """
    consultadas = cenario["variaveis_consultadas"]
    resultados = motor.consultar(consultadas, cenario["evidencias"])
    resultados_base = motor.consultar(consultadas, cenario["evidencias_base"])

    falhas = verificar_condicoes(resultados, resultados_base, cenario["resultados_esperados"])

    # Expectativa intuitiva que a rede não confirma (limitação documentada): ela deve falhar
    nao_confirmada = cenario.get("expectativa_nao_confirmada")
    divergencias = verificar_condicoes(resultados, resultados_base, nao_confirmada) if nao_confirmada else []
    if nao_confirmada and not divergencias:
        falhas.append("'expectativa_nao_confirmada' foi confirmada; a limitação documentada não existe mais.")

    return {
        "resultados": resultados,
        "resultados_base": resultados_base,
        "falhas": falhas,
        "divergencias": divergencias,
    }


def imprimir_cenario(numero: int, cenario: Dict[str, Any], execucao: Dict[str, Any]) -> None:
    """
    Exibe no terminal a pergunta do cenário, a variação de cada probabilidade
    em relação à situação de referência e o resultado das condições de teste.
    """
    print(f"--- {numero}. {cenario['titulo'].upper()} ---")
    print(f"  Modalidade : {cenario['modalidade']}")
    print(f"  Pergunta   : {cenario['pergunta']}")
    print(f"  Evidências : {cenario['evidencias']}")
    print(f"  Base       : {cenario['evidencias_base']}")

    for var, dist in execucao["resultados"].items():
        base = execucao["resultados_base"][var]
        print(f"\n  P({var}): base -> evidências (variação)")
        for estado, p in dist.items():
            print(f"    {estado:<13} {base[estado]:.4f} -> {p:.4f} ({p - base[estado]:+.4f})")

    print(f"\n  Condições de teste: {'REPROVADO' if execucao['falhas'] else 'APROVADO'}")
    for falha in execucao["falhas"]:
        print(f"    - {falha}")
    if execucao["divergencias"]:
        print("  Expectativa não confirmada (limitação do modelo):")
        for divergencia in execucao["divergencias"]:
            print(f"    - {divergencia}")
    print(f"  Interpretação: {cenario['interpretacao']}\n")


if __name__ == "__main__":
    motor = MotorInferencia()
    total_falhas = 0

    for numero, cenario in enumerate(motor.especificacao["cenarios"], start=1):
        execucao = executar_cenario(motor, cenario)
        imprimir_cenario(numero, cenario, execucao)
        total_falhas += len(execucao["falhas"])

    # Código de saída não nulo permite usar o script em verificações automatizadas
    if total_falhas:
        raise SystemExit(f"{total_falhas} condição(ões) de teste não atendida(s).")
