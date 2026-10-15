"""
report.py (áudio)
-----------------
Relatório automático especializado da análise de áudio (Fase 4).

Combina a transcrição e a análise de sinais psicológicos num relatório
estruturado (para a fusão) e num texto legível (para o alerta à equipe).
É triagem de apoio — não diagnostica.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .analyze import AnaliseAudio, AVISO
from .transcribe import Transcricao


@dataclass
class RelatorioAudio:
    fonte: str
    backend_transcricao: str
    duracao_s: float
    sinais: List[str]
    nivel_alerta: str
    detalhes: Dict = field(default_factory=dict)
    resumo_llm: str = ""

    def to_dict(self) -> Dict:
        return {
            "fonte": self.fonte,
            "backend_transcricao": self.backend_transcricao,
            "duracao_s": self.duracao_s,
            "sinais": self.sinais,
            "nivel_alerta": self.nivel_alerta,
            "detalhes": self.detalhes,
            "resumo_llm": self.resumo_llm,
        }


def gerar_relatorio(transc: Transcricao, analise: AnaliseAudio) -> RelatorioAudio:
    # Sinais de triagem = léxico + acústicos (ambos vão para a fusão).
    sinais = [s.rotulo for s in analise.sinais]
    sinais += list(analise.acustica.get("sinais_acusticos", []))
    return RelatorioAudio(
        fonte=transc.fonte,
        backend_transcricao=transc.backend,
        duracao_s=transc.duracao_s,
        sinais=sinais,
        nivel_alerta=analise.nivel_alerta,
        detalhes=analise.to_dict(),
        resumo_llm=analise.resumo_llm,
    )


def formatar_texto(rel: RelatorioAudio) -> str:
    d = rel.detalhes
    linhas = [
        "RELATÓRIO AUTOMÁTICO — ANÁLISE DE ÁUDIO DE CONSULTA",
        f"Fonte: {rel.fonte}",
        f"Transcrição via: {rel.backend_transcricao} | duração: {rel.duracao_s}s",
        f"Nível de alerta: {rel.nivel_alerta.upper()}",
        f"Sinais de triagem: {', '.join(rel.sinais) or 'nenhum'}",
    ]
    for s in d.get("sinais", []):
        linhas.append(f"  - {s['rotulo']}: termos {s['termos']} (intensidade {s['intensidade']})")
    # Sentimento + termos-chave (Azure Text Analytics ou fallback léxico)
    if d.get("sentimento"):
        linhas.append(f"Sentimento ({d.get('backend_texto','lexico')}): {d['sentimento']}"
                      + (f" | termos-chave: {', '.join(d.get('termos_chave', [])[:6])}"
                         if d.get("termos_chave") else ""))
    # Características acústicas
    ac = d.get("acustica") or {}
    if ac.get("disponivel"):
        linhas.append(
            f"Acústica: energia={ac.get('energia_rms')} pitch={ac.get('pitch_medio_hz')}Hz "
            f"pausas={ac.get('proporcao_pausas')} | sinais: "
            f"{', '.join(ac.get('sinais_acusticos', [])) or 'nenhum'}"
        )
    if rel.resumo_llm:
        linhas.append(f"Resumo: {rel.resumo_llm}")
    linhas.append(f"Observação: {AVISO}")
    return "\n".join(linhas)
