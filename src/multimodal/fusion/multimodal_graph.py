"""
multimodal_graph.py
-------------------
Camada de FUSÃO multimodal (Fase 4), com LangGraph.

Recebe os relatórios estruturados das modalidades — vídeo (YOLOv8: instrumentos
e postura), áudio (Azure/Whisper + análise psicológica/acústica) e sinais vitais
(detecção de anomalias) — funde os achados, decide o nível de risco combinado e
dispara o alerta para a equipe especializada, registrando tudo na auditoria.

É o mesmo padrão de fluxo da Fase 3 (triagem → decisão → alerta → auditoria),
agora coordenando múltiplas modalidades. Usa LangGraph quando disponível e cai
num runner Python puro caso contrário (roda em qualquer máquina).

Grafo:

    receber_video → receber_audio → receber_vitais → receber_postura
        → fundir → (risco?) → emitir_alerta | registrar_ok → consolidar
"""

from __future__ import annotations

from typing import Dict, List, Optional, TypedDict

try:  # LangGraph é opcional; há fallback abaixo.
    from langgraph.graph import StateGraph, END
    _TEM_LANGGRAPH = True
except Exception:  # pragma: no cover
    StateGraph = None
    END = "END"
    _TEM_LANGGRAPH = False

# Auditoria reaproveitada da Fase 3 (import defensivo).
try:
    from src.assistant.safety.audit_log import AuditLogger, AuditEvent
except Exception:  # pragma: no cover
    AuditLogger = None
    AuditEvent = None

_NIVEL_ORDEM = {"verde": 0, "amarelo": 1, "vermelho": 2}
_ORDEM_NIVEL = {v: k for k, v in _NIVEL_ORDEM.items()}

# Modalidades fundidas: (chave do relatório, chave do nível, rótulo, campo de achados)
_MODALIDADES = [
    ("relatorio_video", "nivel_video", "vídeo", "anomalias"),
    ("relatorio_audio", "nivel_audio", "áudio", "sinais"),
    ("relatorio_vitais", "nivel_vitais", "sinais vitais", "sinais"),
    ("relatorio_postura", "nivel_postura", "postura", "sinais"),
    ("relatorio_prescricoes", "nivel_prescricoes", "prescrições", "sinais"),
]


class EstadoFusao(TypedDict, total=False):
    paciente_id: str
    relatorio_video: Dict
    relatorio_audio: Dict
    relatorio_vitais: Dict
    relatorio_postura: Dict
    relatorio_prescricoes: Dict
    nivel_video: str
    nivel_audio: str
    nivel_vitais: str
    nivel_postura: str
    nivel_prescricoes: str
    nivel_final: str
    achados: List[str]
    alerta: Optional[str]
    resumo_final: str
    trilha: List[str]


class FusaoMultimodal:
    """Funde os relatórios das modalidades e emite o alerta consolidado."""

    def __init__(self, auditor=None):
        self.auditor = auditor or (AuditLogger() if AuditLogger else None)
        self.app = self._compilar()

    # ---- nós de recepção (um por modalidade) ----------------------------- #
    def _receber(self, estado: EstadoFusao, rel_key: str, nivel_key: str, nome: str) -> EstadoFusao:
        rel = estado.get(rel_key) or {}
        trilha = estado.get("trilha", []) + [f"receber_{nivel_key.split('_')[1]}"]
        return {**estado, nivel_key: rel.get("nivel_alerta", "verde"), "trilha": trilha}

    def _no_receber_video(self, e): return self._receber(e, "relatorio_video", "nivel_video", "vídeo")
    def _no_receber_audio(self, e): return self._receber(e, "relatorio_audio", "nivel_audio", "áudio")
    def _no_receber_vitais(self, e): return self._receber(e, "relatorio_vitais", "nivel_vitais", "sinais vitais")
    def _no_receber_postura(self, e): return self._receber(e, "relatorio_postura", "nivel_postura", "postura")
    def _no_receber_prescricoes(self, e): return self._receber(e, "relatorio_prescricoes", "nivel_prescricoes", "prescrições")

    def _no_fundir(self, estado: EstadoFusao) -> EstadoFusao:
        # Fusão: o maior nível entre TODAS as modalidades prevalece.
        niveis = [_NIVEL_ORDEM[estado.get(nk, "verde")] for _, nk, _, _ in _MODALIDADES]
        final = _ORDEM_NIVEL[max(niveis)]

        achados: List[str] = []
        for rel_key, _, rotulo, campo in _MODALIDADES:
            rel = estado.get(rel_key) or {}
            for a in rel.get(campo, []):
                achados.append(f"[{rotulo}] {a}")

        trilha = estado.get("trilha", []) + ["fundir"]
        return {**estado, "nivel_final": final, "achados": achados, "trilha": trilha}

    def _no_emitir_alerta(self, estado: EstadoFusao) -> EstadoFusao:
        pid = estado.get("paciente_id", "N/D")
        nivel = estado.get("nivel_final", "amarelo")
        achados = estado.get("achados", [])
        alerta = (
            f"🚨 ALERTA MULTIMODAL — paciente {pid} | risco {nivel.upper()}. "
            f"Acionar equipe especializada. Achados: "
            + ("; ".join(achados) if achados else "sinais combinados das modalidades.")
        )
        trilha = estado.get("trilha", []) + ["emitir_alerta"]
        self._auditar(estado, alerta)
        return {**estado, "alerta": alerta, "trilha": trilha}

    def _no_registrar_ok(self, estado: EstadoFusao) -> EstadoFusao:
        trilha = estado.get("trilha", []) + ["registrar_ok"]
        self._auditar(estado, "Sem risco relevante nas modalidades analisadas.")
        return {**estado, "trilha": trilha}

    def _no_consolidar(self, estado: EstadoFusao) -> EstadoFusao:
        niveis_txt = " | ".join(
            f"{rotulo}: {estado.get(nk, 'verde')}" for _, nk, rotulo, _ in _MODALIDADES
        )
        partes = [
            f"Paciente {estado.get('paciente_id','N/D')} | risco combinado: "
            f"{estado.get('nivel_final','verde').upper()}",
            niveis_txt,
        ]
        if estado.get("achados"):
            partes.append("Achados:\n- " + "\n- ".join(estado["achados"]))
        if estado.get("alerta"):
            partes.append(estado["alerta"])
        partes.append("Trilha: " + " → ".join(estado.get("trilha", []) + ["consolidar"]))
        partes.append(
            "Apoio à decisão multimodal — validação humana obrigatória; conduta "
            "é da equipe especializada."
        )
        return {**estado, "resumo_final": "\n\n".join(partes),
                "trilha": estado.get("trilha", []) + ["consolidar"]}

    # ---- auditoria ------------------------------------------------------- #
    def _auditar(self, estado: EstadoFusao, mensagem: str) -> None:
        if not (self.auditor and AuditEvent):
            return
        try:
            self.auditor.registrar(AuditEvent(
                pergunta=f"[fusão multimodal] paciente {estado.get('paciente_id','N/D')}",
                resposta=mensagem,
                backend_llm="multimodal",
                fontes=estado.get("achados", []),
                guardrail_categorias=[],
            ))
        except Exception:
            pass

    # ---- roteamento ------------------------------------------------------ #
    @staticmethod
    def _rota(estado: EstadoFusao) -> str:
        return "emitir_alerta" if estado.get("nivel_final") in ("amarelo", "vermelho") else "registrar_ok"

    # ---- compilação ------------------------------------------------------ #
    def _compilar(self):
        if not _TEM_LANGGRAPH:
            return None  # usa o runner Python puro em executar()
        g = StateGraph(EstadoFusao)
        g.add_node("receber_video", self._no_receber_video)
        g.add_node("receber_audio", self._no_receber_audio)
        g.add_node("receber_vitais", self._no_receber_vitais)
        g.add_node("receber_postura", self._no_receber_postura)
        g.add_node("receber_prescricoes", self._no_receber_prescricoes)
        g.add_node("fundir", self._no_fundir)
        g.add_node("emitir_alerta", self._no_emitir_alerta)
        g.add_node("registrar_ok", self._no_registrar_ok)
        g.add_node("consolidar", self._no_consolidar)
        g.set_entry_point("receber_video")
        g.add_edge("receber_video", "receber_audio")
        g.add_edge("receber_audio", "receber_vitais")
        g.add_edge("receber_vitais", "receber_postura")
        g.add_edge("receber_postura", "receber_prescricoes")
        g.add_edge("receber_prescricoes", "fundir")
        g.add_conditional_edges("fundir", self._rota,
                                {"emitir_alerta": "emitir_alerta", "registrar_ok": "registrar_ok"})
        g.add_edge("emitir_alerta", "consolidar")
        g.add_edge("registrar_ok", "consolidar")
        g.add_edge("consolidar", END)
        return g.compile()

    def _executar_puro(self, estado: EstadoFusao) -> EstadoFusao:
        """Fallback sem LangGraph: executa os nós na mesma ordem."""
        estado = self._no_receber_video(estado)
        estado = self._no_receber_audio(estado)
        estado = self._no_receber_vitais(estado)
        estado = self._no_receber_postura(estado)
        estado = self._no_receber_prescricoes(estado)
        estado = self._no_fundir(estado)
        if self._rota(estado) == "emitir_alerta":
            estado = self._no_emitir_alerta(estado)
        else:
            estado = self._no_registrar_ok(estado)
        return self._no_consolidar(estado)

    def executar(self, relatorio_video: Dict, relatorio_audio: Dict,
                 paciente_id: str = "N/D",
                 relatorio_vitais: Optional[Dict] = None,
                 relatorio_postura: Optional[Dict] = None,
                 relatorio_prescricoes: Optional[Dict] = None) -> EstadoFusao:
        entrada: EstadoFusao = {
            "paciente_id": paciente_id,
            "relatorio_video": relatorio_video,
            "relatorio_audio": relatorio_audio,
            "relatorio_vitais": relatorio_vitais or {},
            "relatorio_postura": relatorio_postura or {},
            "relatorio_prescricoes": relatorio_prescricoes or {},
            "trilha": [],
        }
        if self.app is not None:
            return self.app.invoke(entrada)
        return self._executar_puro(entrada)
