"""
report.py
---------
Relatório automático especializado a partir da análise de vídeo (Fase 4).

Converte as detecções do YOLOv8 em achados clínicos estruturados e aplica regras
simples de anomalia/desvio, gerando: (1) um dicionário estruturado para a camada
de fusão e (2) um texto legível para o relatório e para o alerta à equipe.

As regras são heurísticas de DEMONSTRAÇÃO e servem de apoio à decisão — nunca
substituem a avaliação da equipe cirúrgica. Em produção, os limiares e o conjunto
de instrumentos esperados viriam do protocolo do procedimento.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .detect import ResultadoVideo

# Instrumentos esperados no procedimento (viria do protocolo, em produção).
INSTRUMENTOS_ESPERADOS = {"bisturi", "pinca", "tesoura", "afastador"}

# Confiança mínima para considerar uma detecção "firme".
CONF_MINIMA = 0.5


@dataclass
class RelatorioVideo:
    procedimento: str
    duracao_s: float
    instrumentos: List[str]
    contagem: Dict[str, int]
    anomalias: List[str] = field(default_factory=list)
    nivel_alerta: str = "verde"           # verde | amarelo | vermelho
    video_anotado: str = ""

    def to_dict(self) -> Dict:
        return {
            "procedimento": self.procedimento,
            "duracao_s": self.duracao_s,
            "instrumentos": self.instrumentos,
            "contagem": self.contagem,
            "anomalias": self.anomalias,
            "nivel_alerta": self.nivel_alerta,
            "video_anotado": self.video_anotado,
        }


def _cobertura_temporal(res: ResultadoVideo) -> float:
    """Fração de frames analisados que tiveram ao menos uma detecção firme."""
    if res.n_frames == 0:
        return 0.0
    frames_com_deteccao = {d.frame for d in res.deteccoes if d.confianca >= CONF_MINIMA}
    return round(len(frames_com_deteccao) / res.n_frames, 3)


def gerar_relatorio(res: ResultadoVideo, procedimento: str = "cirurgia ginecológica") -> RelatorioVideo:
    """Aplica as regras de anomalia e monta o relatório especializado."""
    firmes = [d for d in res.deteccoes if d.confianca >= CONF_MINIMA]
    contagem: Dict[str, int] = {}
    for d in firmes:
        contagem[d.classe] = contagem.get(d.classe, 0) + 1
    instrumentos = sorted(contagem.keys())

    anomalias: List[str] = []

    # Regra 1: instrumento fora do conjunto esperado do procedimento.
    inesperados = [c for c in instrumentos if c not in INSTRUMENTOS_ESPERADOS]
    for c in inesperados:
        anomalias.append(f"Instrumento não previsto no protocolo detectado: {c}.")

    # Regra 2: baixa cobertura temporal (possível oclusão/campo obstruído).
    cobertura = _cobertura_temporal(res)
    if cobertura < 0.4:
        anomalias.append(
            f"Baixa cobertura temporal de instrumentos ({cobertura:.0%}) — "
            "possível obstrução do campo ou pausa não documentada."
        )

    # Regra 3: contagem final de instrumentos (proxy de checagem de contagem).
    total = sum(contagem.values())
    if total == 0:
        anomalias.append("Nenhum instrumento detectado no procedimento — revisar captação.")

    # Nível de alerta a partir das anomalias.
    if inesperados:
        nivel = "vermelho"
    elif anomalias:
        nivel = "amarelo"
    else:
        nivel = "verde"

    return RelatorioVideo(
        procedimento=procedimento,
        duracao_s=res.duracao_s,
        instrumentos=instrumentos,
        contagem=contagem,
        anomalias=anomalias,
        nivel_alerta=nivel,
        video_anotado=res.video_anotado or "",
    )


def formatar_texto(rel: RelatorioVideo) -> str:
    """Gera o texto legível do relatório especializado."""
    linhas = [
        "RELATÓRIO AUTOMÁTICO — ANÁLISE DE VÍDEO CIRÚRGICO",
        f"Procedimento: {rel.procedimento}",
        f"Duração analisada: {rel.duracao_s}s",
        f"Instrumentos identificados: {', '.join(rel.instrumentos) or 'nenhum'}",
        f"Contagem por instrumento: {rel.contagem or '—'}",
        f"Nível de alerta: {rel.nivel_alerta.upper()}",
    ]
    if rel.anomalias:
        linhas.append("Anomalias/desvios:")
        linhas += [f"  - {a}" for a in rel.anomalias]
    else:
        linhas.append("Anomalias/desvios: nenhuma detectada.")
    linhas.append(
        "Observação: apoio à decisão baseado em visão computacional; "
        "a conduta é da equipe cirúrgica responsável."
    )
    return "\n".join(linhas)
