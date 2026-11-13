from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pgmpy.inference import VariableElimination

try:
    from src.rede import CAMINHO_PADRAO_JSON, construir_rede
except ImportError:
    from rede import CAMINHO_PADRAO_JSON, construir_rede


class MotorInferencia:
    """
    Encapsula a Rede Bayesiana e o algoritmo de Eliminação de Variáveis,
    fornecendo validação estrita de entradas e saídas normalizadas.
    """

    def __init__(self, caminho_json: str | Path = CAMINHO_PADRAO_JSON) -> None:
        self.caminho_json = Path(caminho_json)
        self.modelo, self.especificacao = construir_rede(self.caminho_json)
        self.eliminador = VariableElimination(self.modelo)

        self.mapa_estados: Dict[str, List[str]] = {
            v["id"]: list(v["estados"]) for v in self.especificacao["variaveis"]
        }
        self.variaveis_validas = set(self.mapa_estados.keys())

    def validar_consulta(
        self,
        variaveis_consulta: List[str],
        evidencias: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Verifica se as variáveis consultadas e as evidências informadas são válidas
        perante o domínio definido no modelo.json. Levanta ValueError em casos inválidos.
        """
        if not variaveis_consulta or not isinstance(variaveis_consulta, (list, tuple)):
            raise ValueError("A lista 'variaveis_consulta' deve conter ao menos uma variável.")

        for var in variaveis_consulta:
            if var not in self.variaveis_validas:
                raise ValueError(
                    f"Variável de consulta '{var}' inexistente na rede. "
                    f"Variáveis válidas: {sorted(self.variaveis_validas)}"
                )

        if len(set(variaveis_consulta)) != len(variaveis_consulta):
            raise ValueError("A lista 'variaveis_consulta' contém variáveis duplicadas.")

        if evidencias is not None:
            if not isinstance(evidencias, dict):
                raise ValueError("O parâmetro 'evidencias' deve ser um dicionário {variavel: estado}.")

            for var_ev, estado_ev in evidencias.items():
                if var_ev not in self.variaveis_validas:
                    raise ValueError(
                        f"Variável de evidência '{var_ev}' não existe na rede. "
                        f"Variáveis válidas: {sorted(self.variaveis_validas)}"
                    )

                estados_permitidos = self.mapa_estados[var_ev]
                if estado_ev not in estados_permitidos:
                    raise ValueError(
                        f"Estado '{estado_ev}' inválido para a variável '{var_ev}'. "
                        f"Estados permitidos: {estados_permitidos}"
                    )

                if var_ev in variaveis_consulta:
                    raise ValueError(
                        f"A variável '{var_ev}' não pode estar simultaneamente em "
                        f"'variaveis_consulta' e em 'evidencias'."
                    )

    def consultar(
        self,
        variaveis_consulta: List[str],
        evidencias: Optional[Dict[str, str]] = None,
        casas_decimais: int = 4,
    ) -> Dict[str, Dict[str, float]]:
        """
        Executa a inferência marginal posterior P(X_i | E = e) para cada variável
        solicitada em 'variaveis_consulta', retornando distribuições normalizadas.
        """
        self.validar_consulta(variaveis_consulta, evidencias)
        ev_limpas = evidencias if evidencias else None

        resultados: Dict[str, Dict[str, float]] = {}

        for var in variaveis_consulta:
            fator = self.eliminador.query(
                variables=[var],
                evidence=ev_limpas,
                show_progress=False,
            )

            valores = np.asarray(fator.values, dtype=float)
            soma = float(np.sum(valores))
            if soma <= 0.0:
                raise RuntimeError(f"Distribuição degenerada (soma=0) ao consultar '{var}'.")

            # Garantir normalização estrita
            valores_norm = valores / soma
            estados = self.mapa_estados[var]

            dist_dict = {
                estado: round(float(prob), casas_decimais)
                for estado, prob in zip(estados, valores_norm)
            }
            resultados[var] = dist_dict

        return resultados

    def consultar_conjunta(
        self,
        variaveis_consulta: List[str],
        evidencias: Optional[Dict[str, str]] = None,
        casas_decimais: int = 4,
    ) -> Dict[str, float]:
        """
        Calcula a distribuição conjunta posterior P(X_1, X_2, ... | E = e) sobre uma
        tupla de variáveis de consulta.
        """
        self.validar_consulta(variaveis_consulta, evidencias)
        ev_limpas = evidencias if evidencias else None

        fator = self.eliminador.query(
            variables=list(variaveis_consulta),
            evidence=ev_limpas,
            show_progress=False,
        )

        valores = np.asarray(fator.values, dtype=float)
        valores_norm = valores / float(np.sum(valores))

        listas_estados = [self.mapa_estados[v] for v in variaveis_consulta]
        grade_indices = np.ndindex(*valores_norm.shape)

        dist_conjunta: Dict[str, float] = {}
        for idx_tupla in grade_indices:
            nomes_comb = "|".join(
                listas_estados[pos][i] for pos, i in enumerate(idx_tupla)
            )
            dist_conjunta[nomes_comb] = round(float(valores_norm[idx_tupla]), casas_decimais)

        return dist_conjunta

    def comparar_evidencias(
        self,
        variavel_consulta: str,
        evidencias_base: Optional[Dict[str, str]],
        evidencias_novas: Dict[str, str],
        casas_decimais: int = 4,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compara a distribuição de uma variável entre um cenário base (ou a priori, se None)
        e um novo conjunto de evidências, calculando a variação (delta) em cada estado.
        """
        dist_base = self.consultar([variavel_consulta], evidencias_base, casas_decimais)[
            variavel_consulta
        ]
        dist_nova = self.consultar([variavel_consulta], evidencias_novas, casas_decimais)[
            variavel_consulta
        ]

        diferencas = {
            estado: round(dist_nova[estado] - dist_base[estado], casas_decimais)
            for estado in self.mapa_estados[variavel_consulta]
        }

        return {
            "base": dist_base,
            "com_novas_evidencias": dist_nova,
            "variacao_delta": diferencas,
        }


if __name__ == "__main__":
    motor = MotorInferencia()

    print("--- 1. DISTRIBUIÇÃO A PRIORI (SEM EVIDÊNCIAS) DE DEMANDA ---")
    priori = motor.consultar(["demanda"])
    print(f"  P(demanda) = {priori['demanda']}\n")

    print("--- 2. COMPARAÇÃO: HORÁRIO DE PICO COM CLIMA IDEAL VS. CHUVA FORTE ---")
    ev_ideal = {
        "temperatura": "Amena",
        "precipitacao": "Nenhuma",
        "vento": "Fraco",
        "horario": "Pico",
        "feriado": "Nao",
        "dia_ativo": "Sim",
    }
    ev_chuva = {
        "temperatura": "Amena",
        "precipitacao": "Chuva",
        "vento": "Forte",
        "horario": "Pico",
        "feriado": "Nao",
        "dia_ativo": "Sim",
    }
    comp = motor.comparar_evidencias("demanda", ev_ideal, ev_chuva)
    print(f"  Clima Ideal : {comp['base']}")
    print(f"  Chuva Forte : {comp['com_novas_evidencias']}")
    print(f"  Delta       : {comp['variacao_delta']}")