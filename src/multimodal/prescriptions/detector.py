"""
detector.py
-----------
Detecção de anomalias na EVOLUÇÃO DE PRESCRIÇÕES (Fase 4).

Analisa o histórico de prescrições de uma paciente ao longo da internação e
sinaliza mudanças inesperadas no tratamento:

  - **Variação brusca de dose**: aumento/redução acima de um limiar relativo
    (padrão ±50%) entre registros consecutivos do mesmo medicamento.
  - **Troca de medicamento**: substituição da mesma classe/indicação por outro
    princípio ativo em curto intervalo.
  - **Suspensão**: medicamento de uso contínuo que deixa de aparecer sem registro
    de desmame.
  - **Reintrodução**: medicamento suspenso que retorna (possível erro de conciliação).

É **determinístico e explicável** — cada anomalia aponta o medicamento, os
registros envolvidos e o motivo. Os dados de demonstração são sintéticos. É apoio
à decisão: toda alteração de prescrição é ato médico e exige validação humana.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .. import config

# Medicamentos de uso contínuo cuja ausência súbita é relevante (demonstração).
_USO_CONTINUO = {"metildopa", "insulina", "levotiroxina", "enoxaparina"}


@dataclass
class AnomaliaPrescricao:
    instante: int
    medicamento: str
    tipo: str            # dose_brusca | troca | suspensao | reintroducao
    motivo: str
    gravidade: str       # amarelo | vermelho


@dataclass
class RelatorioPrescricoes:
    paciente_id: str
    n_registros: int
    anomalias: List[AnomaliaPrescricao] = field(default_factory=list)
    nivel_alerta: str = "verde"

    def to_dict(self) -> Dict:
        return {
            "paciente_id": self.paciente_id,
            "n_registros": self.n_registros,
            "nivel_alerta": self.nivel_alerta,
            # compat. com a fusão: descrições curtas por anomalia
            "sinais": [f"{a.medicamento}: {a.motivo}" for a in self.anomalias],
            "anomalias": [
                {"instante": a.instante, "medicamento": a.medicamento,
                 "tipo": a.tipo, "motivo": a.motivo, "gravidade": a.gravidade}
                for a in self.anomalias
            ],
        }


# --------------------------------------------------------------------------- #
# Dados sintéticos de demonstração
# --------------------------------------------------------------------------- #
def gerar_historico_demo(paciente_id: str = "PAC-0007", com_anomalias: bool = True) -> Path:
    """
    Gera um histórico de prescrições sintético (CSV) e o salva.
    Colunas: instante, medicamento, dose, unidade, indicacao.
    """
    config.ensure_dirs()
    linhas = [
        {"instante": 0, "medicamento": "metildopa", "dose": 250, "unidade": "mg", "indicacao": "hipertensao"},
        {"instante": 1, "medicamento": "metildopa", "dose": 250, "unidade": "mg", "indicacao": "hipertensao"},
        {"instante": 2, "medicamento": "metildopa", "dose": 250, "unidade": "mg", "indicacao": "hipertensao"},
        {"instante": 3, "medicamento": "metildopa", "dose": 250, "unidade": "mg", "indicacao": "hipertensao"},
    ]
    if com_anomalias:
        # variação brusca de dose (250 -> 750 mg = +200%)
        linhas.append({"instante": 4, "medicamento": "metildopa", "dose": 750, "unidade": "mg", "indicacao": "hipertensao"})
        # troca de medicamento para a mesma indicação (metildopa -> nifedipino)
        linhas.append({"instante": 5, "medicamento": "nifedipino", "dose": 20, "unidade": "mg", "indicacao": "hipertensao"})
        # suspensão: metildopa (uso contínuo) não aparece mais a partir daqui
    else:
        linhas.append({"instante": 4, "medicamento": "metildopa", "dose": 250, "unidade": "mg", "indicacao": "hipertensao"})
        linhas.append({"instante": 5, "medicamento": "metildopa", "dose": 250, "unidade": "mg", "indicacao": "hipertensao"})

    caminho = config.PRESCRIPTIONS_DATA_DIR / f"prescricoes_{paciente_id}.csv"
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["instante", "medicamento", "dose", "unidade", "indicacao"])
        w.writeheader()
        w.writerows(linhas)
    return caminho


def _ler_csv(caminho: Path) -> List[Dict]:
    with open(caminho, encoding="utf-8") as f:
        out = []
        for row in csv.DictReader(f):
            row["instante"] = int(row["instante"])
            try:
                row["dose"] = float(row["dose"])
            except (ValueError, KeyError):
                row["dose"] = None
            out.append(row)
        return out


# --------------------------------------------------------------------------- #
# Detecção
# --------------------------------------------------------------------------- #
def analisar(caminho_csv: str, paciente_id: str = "N/D",
             variacao_brusca: Optional[float] = None) -> RelatorioPrescricoes:
    limiar = variacao_brusca if variacao_brusca is not None else config.PRESC_VARIACAO_BRUSCA
    regs = _ler_csv(Path(caminho_csv))
    anomalias: List[AnomaliaPrescricao] = []

    # última dose por medicamento e último instante visto (p/ dose brusca)
    ultima_dose: Dict[str, float] = {}
    # medicamentos ativos por indicação (p/ troca) e instante da última aparição
    ativo_por_indicacao: Dict[str, str] = {}
    ultimo_instante_med: Dict[str, int] = {}

    instantes = sorted({r["instante"] for r in regs})
    for r in regs:
        med, dose, ind, t = r["medicamento"], r["dose"], r.get("indicacao", ""), r["instante"]

        # 1) variação brusca de dose (mesmo medicamento)
        if dose is not None and med in ultima_dose and ultima_dose[med]:
            delta = abs(dose - ultima_dose[med]) / ultima_dose[med]
            if delta >= limiar:
                sentido = "aumento" if dose > ultima_dose[med] else "redução"
                grav = "vermelho" if delta >= 1.0 else "amarelo"
                anomalias.append(AnomaliaPrescricao(
                    t, med, "dose_brusca",
                    f"{sentido} brusco de dose ({ultima_dose[med]:.0f}→{dose:.0f} {r.get('unidade','')}, {delta:+.0%})",
                    grav))
        if dose is not None:
            ultima_dose[med] = dose

        # 2) troca de medicamento para a mesma indicação
        if ind and ind in ativo_por_indicacao and ativo_por_indicacao[ind] != med:
            anomalias.append(AnomaliaPrescricao(
                t, med, "troca",
                f"troca de medicamento para '{ind}' ({ativo_por_indicacao[ind]}→{med})",
                "amarelo"))
            # medicamento anterior saiu de cena → possível suspensão checada abaixo
        if ind:
            ativo_por_indicacao[ind] = med

        # 3) reintrodução: medicamento volta depois de ausência longa
        if med in ultimo_instante_med and t - ultimo_instante_med[med] >= 3:
            anomalias.append(AnomaliaPrescricao(
                t, med, "reintroducao",
                f"reintrodução após ausência (último em t={ultimo_instante_med[med]})",
                "amarelo"))
        ultimo_instante_med[med] = t

    # 4) suspensão: medicamento de uso contínuo que some antes do fim da série
    fim = instantes[-1] if instantes else 0
    for med, t_ultimo in ultimo_instante_med.items():
        if med in _USO_CONTINUO and t_ultimo < fim:
            anomalias.append(AnomaliaPrescricao(
                fim, med, "suspensao",
                f"suspensão de medicamento de uso contínuo (último em t={t_ultimo}, sem registro de desmame)",
                "vermelho"))

    anomalias.sort(key=lambda a: a.instante)
    if any(a.gravidade == "vermelho" for a in anomalias):
        nivel = "vermelho"
    elif anomalias:
        nivel = "amarelo"
    else:
        nivel = "verde"

    return RelatorioPrescricoes(
        paciente_id=paciente_id, n_registros=len(regs),
        anomalias=anomalias, nivel_alerta=nivel,
    )


def formatar_texto(rel: RelatorioPrescricoes) -> str:
    linhas = [
        "RELATÓRIO AUTOMÁTICO — EVOLUÇÃO DE PRESCRIÇÕES",
        f"Paciente: {rel.paciente_id} | registros: {rel.n_registros}",
        f"Nível de alerta: {rel.nivel_alerta.upper()}",
    ]
    if rel.anomalias:
        linhas.append(f"Anomalias ({len(rel.anomalias)}):")
        for a in rel.anomalias:
            linhas.append(f"  - t={a.instante} {a.medicamento} | {a.motivo} [{a.tipo}/{a.gravidade}]")
    else:
        linhas.append("Nenhuma anomalia detectada na evolução das prescrições.")
    linhas.append("Observação: toda alteração de prescrição é ato médico e exige validação humana.")
    return "\n".join(linhas)


if __name__ == "__main__":
    csv_path = gerar_historico_demo()
    rel = analisar(str(csv_path), paciente_id="PAC-0007")
    print(formatar_texto(rel))
