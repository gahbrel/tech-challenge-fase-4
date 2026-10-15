"""
detector.py
-----------
Detecção de anomalias em sinais vitais específicos da saúde materna (Fase 4).

Monitora séries temporais de sinais vitais de gestantes — pressão arterial
(sistólica/diastólica), batimentos cardíacos fetais, frequência cardíaca materna
e saturação — e sinaliza anomalias combinando:

  1. **Regras clínicas** (determinísticas, explicáveis): faixas de referência
     obstétricas (ex.: PA >= 140/90 → hipertensão gestacional; BCF < 110 ou
     > 160 → fora da faixa normal).
  2. **Modelo de detecção de anomalias** (Isolation Forest, scikit-learn) ou
     z-score como alternativa — captura desvios estatísticos mesmo dentro das
     faixas, úteis para monitoramento preventivo.

As séries de demonstração são sintéticas (com anomalias injetadas). Em produção,
os dados viriam de monitores/wearables. É apoio à decisão — exige validação.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .. import config

# Faixas de referência (apoio; simplificadas para demonstração).
_REGRAS = {
    "pressao_sistolica": {"max_amarelo": 140, "max_vermelho": 160, "unidade": "mmHg"},
    "pressao_diastolica": {"max_amarelo": 90, "max_vermelho": 110, "unidade": "mmHg"},
    "batimento_fetal": {"min": 110, "max": 160, "min_vermelho": 100, "unidade": "bpm"},
    "freq_cardiaca_materna": {"min": 50, "max": 120, "unidade": "bpm"},
    "saturacao": {"min": 94, "min_vermelho": 90, "unidade": "%"},
}
_COLUNAS = list(_REGRAS.keys())


@dataclass
class AnomaliaVital:
    instante: int
    sinal: str
    valor: float
    motivo: str
    origem: str          # "regra" | "modelo"
    gravidade: str       # "amarelo" | "vermelho"


@dataclass
class RelatorioVitais:
    paciente_id: str
    n_amostras: int
    anomalias: List[AnomaliaVital] = field(default_factory=list)
    nivel_alerta: str = "verde"
    metodo: str = ""

    def resumo_sinais(self) -> List[str]:
        return sorted({a.sinal for a in self.anomalias})

    def to_dict(self) -> Dict:
        return {
            "paciente_id": self.paciente_id,
            "n_amostras": self.n_amostras,
            "nivel_alerta": self.nivel_alerta,
            "metodo": self.metodo,
            # compat. com a fusão: descrições únicas por (sinal, motivo)
            "sinais": sorted({f"{a.sinal} ({a.motivo})" for a in self.anomalias}),
            "anomalias": [
                {"instante": a.instante, "sinal": a.sinal, "valor": a.valor,
                 "motivo": a.motivo, "origem": a.origem, "gravidade": a.gravidade}
                for a in self.anomalias
            ],
        }


# --------------------------------------------------------------------------- #
# Dados sintéticos de demonstração
# --------------------------------------------------------------------------- #
def gerar_serie_demo(paciente_id: str = "PAC-0007", n: int = 60, seed: int = 7,
                     com_anomalias: bool = True) -> Path:
    """Gera uma série temporal sintética de sinais vitais e salva em CSV."""
    rng = random.Random(seed)
    config.ensure_dirs()
    linhas = []
    for t in range(n):
        linha = {
            "instante": t,
            "pressao_sistolica": round(rng.gauss(118, 6), 1),
            "pressao_diastolica": round(rng.gauss(76, 5), 1),
            "batimento_fetal": round(rng.gauss(140, 7), 1),
            "freq_cardiaca_materna": round(rng.gauss(82, 6), 1),
            "saturacao": round(rng.gauss(97, 1), 1),
        }
        linhas.append(linha)

    if com_anomalias:
        # pico hipertensivo (possível pré-eclâmpsia) por volta de t=40
        for t in range(40, 44):
            linhas[t]["pressao_sistolica"] = round(rng.gauss(165, 4), 1)
            linhas[t]["pressao_diastolica"] = round(rng.gauss(112, 3), 1)
        # bradicardia fetal em t=52
        linhas[52]["batimento_fetal"] = 98.0
        # leve dessaturação em t=20 (anomalia estatística, ainda > limiar)
        linhas[20]["saturacao"] = 94.5

    caminho = config.VITALS_DATA_DIR / f"sinais_vitais_{paciente_id}.csv"
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["instante", *_COLUNAS])
        w.writeheader()
        w.writerows(linhas)
    return caminho


def _ler_csv(caminho: Path) -> List[Dict[str, float]]:
    with open(caminho, encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


# --------------------------------------------------------------------------- #
# Regras clínicas
# --------------------------------------------------------------------------- #
def _checar_regras(amostras: List[Dict[str, float]]) -> List[AnomaliaVital]:
    anomalias: List[AnomaliaVital] = []
    for row in amostras:
        t = int(row["instante"])
        ps, pd = row["pressao_sistolica"], row["pressao_diastolica"]
        if ps >= _REGRAS["pressao_sistolica"]["max_vermelho"] or pd >= _REGRAS["pressao_diastolica"]["max_vermelho"]:
            anomalias.append(AnomaliaVital(t, "pressao_arterial", f"{ps:.0f}/{pd:.0f}",
                             "PA muito elevada (possível pré-eclâmpsia)", "regra", "vermelho"))
        elif ps >= _REGRAS["pressao_sistolica"]["max_amarelo"] or pd >= _REGRAS["pressao_diastolica"]["max_amarelo"]:
            anomalias.append(AnomaliaVital(t, "pressao_arterial", f"{ps:.0f}/{pd:.0f}",
                             "PA elevada (hipertensão gestacional)", "regra", "amarelo"))
        bcf = row["batimento_fetal"]
        if bcf < _REGRAS["batimento_fetal"]["min_vermelho"]:
            anomalias.append(AnomaliaVital(t, "batimento_fetal", bcf,
                             "bradicardia fetal", "regra", "vermelho"))
        elif bcf < _REGRAS["batimento_fetal"]["min"] or bcf > _REGRAS["batimento_fetal"]["max"]:
            anomalias.append(AnomaliaVital(t, "batimento_fetal", bcf,
                             "BCF fora da faixa normal", "regra", "amarelo"))
        sat = row["saturacao"]
        if sat < _REGRAS["saturacao"]["min_vermelho"]:
            anomalias.append(AnomaliaVital(t, "saturacao", sat, "dessaturação grave", "regra", "vermelho"))
    return anomalias


# --------------------------------------------------------------------------- #
# Modelo de detecção de anomalias
# --------------------------------------------------------------------------- #
def _checar_modelo(amostras: List[Dict[str, float]], metodo: str) -> List[AnomaliaVital]:
    import numpy as np
    X = np.array([[row[c] for c in _COLUNAS] for row in amostras], dtype=float)
    anomalias: List[AnomaliaVital] = []

    if metodo == "isolation_forest":
        from sklearn.ensemble import IsolationForest
        modelo = IsolationForest(contamination=0.08, random_state=42)
        pred = modelo.fit_predict(X)  # -1 = anomalia
        idxs = [i for i, p in enumerate(pred) if p == -1]
    else:  # zscore
        mu, sigma = X.mean(axis=0), X.std(axis=0) + 1e-9
        z = np.abs((X - mu) / sigma)
        idxs = [i for i in range(len(X)) if z[i].max() > 3.0]

    for i in idxs:
        t = int(amostras[i]["instante"])
        # identifica o sinal mais desviante nessa amostra
        mu, sigma = X.mean(axis=0), X.std(axis=0) + 1e-9
        z = np.abs((X[i] - mu) / sigma)
        c = _COLUNAS[int(z.argmax())]
        anomalias.append(AnomaliaVital(t, c, round(float(amostras[i][c]), 1),
                         "desvio estatístico detectado pelo modelo", "modelo", "amarelo"))
    return anomalias


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def analisar(caminho_csv: str, paciente_id: str = "N/D",
             metodo: Optional[str] = None) -> RelatorioVitais:
    metodo = metodo or config.VITALS_METHOD
    amostras = _ler_csv(Path(caminho_csv))

    anomalias = _checar_regras(amostras)
    # evita duplicar instante/sinal já pego por regra
    ja = {(a.instante, a.sinal) for a in anomalias}
    for a in _checar_modelo(amostras, metodo):
        if (a.instante, a.sinal) not in ja:
            anomalias.append(a)

    anomalias.sort(key=lambda a: a.instante)
    if any(a.gravidade == "vermelho" for a in anomalias):
        nivel = "vermelho"
    elif anomalias:
        nivel = "amarelo"
    else:
        nivel = "verde"

    return RelatorioVitais(
        paciente_id=paciente_id, n_amostras=len(amostras),
        anomalias=anomalias, nivel_alerta=nivel, metodo=metodo,
    )


def formatar_texto(rel: RelatorioVitais) -> str:
    linhas = [
        "RELATÓRIO AUTOMÁTICO — SINAIS VITAIS (saúde materna)",
        f"Paciente: {rel.paciente_id} | amostras: {rel.n_amostras} | método: {rel.metodo}",
        f"Nível de alerta: {rel.nivel_alerta.upper()}",
    ]
    if rel.anomalias:
        linhas.append(f"Anomalias ({len(rel.anomalias)}):")
        for a in rel.anomalias[:12]:
            linhas.append(f"  - t={a.instante} {a.sinal}={a.valor} | {a.motivo} "
                          f"[{a.origem}/{a.gravidade}]")
    else:
        linhas.append("Nenhuma anomalia detectada.")
    linhas.append("Observação: apoio à decisão; validação da equipe obstétrica obrigatória.")
    return "\n".join(linhas)


if __name__ == "__main__":
    csv_path = gerar_serie_demo()
    rel = analisar(str(csv_path), paciente_id="PAC-0007")
    print(formatar_texto(rel))
