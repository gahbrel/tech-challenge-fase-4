"""
multimodal_graph.py
-------------------
Camada de FUSÃO multimodal (Fase 4), com LangGraph.

Recebe os relatórios estruturados de vídeo (YOLOv8) e áudio (Azure/Whisper +
análise psicológica), funde os achados, decide o nível de risco combinado e
dispara o alerta para a equipe especializada — registrando tudo na auditoria.

É o mesmo padrão de fluxo da Fase 3 (triagem → decisão → alerta → auditoria),
agora coordenando múltiplas modalidades. Segue o projeto: usa LangGraph quando
disponível e cai num runner Python puro caso contrário (roda em qualquer máquina).

Grafo:

    receber_video → receber_audio → fundir → (risco?) → emitir_alerta | registrar_ok → consolidar
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


class EstadoFusao(TypedDict, total=False):
    paciente_id: str
    relatorio_video: Dict
    relatorio_audio: Dict
    nivel_video: str
    nivel_audio: str
    nivel_final: str
    achados: List[str]
    alerta: Optional[str]
    resumo_final: str
    trilha: List[str]


class FusaoMultimodal:
    """Funde os relatórios de vídeo e áudio e emite o alerta consolidado."""

    def __init__(self, auditor=None):
        self.auditor = auditor or (AuditLogger() if AuditLogger else None)
        self.app = self._compilar()

    # ---- nós ------------------------------------------------------------- #
    def _no_receber_video(self, estado: EstadoFusao) -> EstadoFusao:
        rv = estado.get("relatorio_video") or {}
        trilha = estado.get("trilha", []) + ["receber_video"]
        return {**estado, "nivel_video": rv.get("nivel_alerta", "verde"), "trilha": trilha}

    def _no_receber_audio(self, estado: EstadoFusao) -> EstadoFusao:
        ra = estado.get("relatorio_audio") or {}
        trilha = estado.get("trilha", []) + ["receber_audio"]
        return {**estado, "nivel_audio": ra.get("nivel_alerta", "verde"), "trilha": trilha}

    def _no_fundir(self, estado: EstadoFusao) -> EstadoFusao:
        nv = estado.get("nivel_video", "verde")
        na = estado.get("nivel_audio", "verde")
        # Fusão: o maior nível entre as modalidades prevalece.
        final = _ORDEM_NIVEL[max(_NIVEL_ORDEM[nv], _NIVEL_ORDEM[na])]

        achados: List[str] = []
        rv = estado.get("relatorio_video") or {}
        ra = estado.get("relatorio_audio") or {}
        for a in rv.get("anomalias", []):
            achados.append(f"[vídeo] {a}")
        for s in ra.get("sinais", []):
            achados.append(f"[áudio] {s}")

        trilha = estado.get("trilha", []) + ["fundir"]
        return {**estado, "nivel_final": final, "achados": achados, "trilha": trilha}

    def _no_emitir_alerta(self, estado: EstadoFusao) -> EstadoFusao:
        pid = estado.get("paciente_id", "N/D")
        nivel = estado.get("nivel_final", "amarelo")
        achados = estado.get("achados", [])
        alerta = (
            f"🚨 ALERTA MULTIMODAL — paciente {pid} | risco {nivel.upper()}. "
            f"Acionar equipe especializada. Achados: "
            + ("; ".join(achados) if achados else "sinais combinados de vídeo/áudio.")
        )
        trilha = estado.get("trilha", []) + ["emitir_alerta"]
        self._auditar(estado, alerta)
        return {**estado, "alerta": alerta, "trilha": trilha}

    def _no_registrar_ok(self, estado: EstadoFusao) -> EstadoFusao:
        trilha = estado.get("trilha", []) + ["registrar_ok"]
        self._auditar(estado, "Sem risco relevante nas modalidades analisadas.")
        return {**estado, "trilha": trilha}

    def _no_consolidar(self, estado: EstadoFusao) -> EstadoFusao:
        partes = [
            f"Paciente {estado.get('paciente_id','N/D')} | risco combinado: "
            f"{estado.get('nivel_final','verde').upper()}",
            f"Vídeo: {estado.get('nivel_video','verde')} | Áudio: {estado.get('nivel_audio','verde')}",
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
        g.add_node("fundir", self._no_fundir)
        g.add_node("emitir_alerta", self._no_emitir_alerta)
        g.add_node("registrar_ok", self._no_registrar_ok)
        g.add_node("consolidar", self._no_consolidar)
        g.set_entry_point("receber_video")
        g.add_edge("receber_video", "receber_audio")
        g.add_edge("receber_audio", "fundir")
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
        estado = self._no_fundir(estado)
        if self._rota(estado) == "emitir_alerta":
            estado = self._no_emitir_alerta(estado)
        else:
            estado = self._no_registrar_ok(estado)
        return self._no_consolidar(estado)

    def executar(self, relatorio_video: Dict, relatorio_audio: Dict,
                 paciente_id: str = "N/D") -> EstadoFusao:
        entrada: EstadoFusao = {
            "paciente_id": paciente_id,
            "relatorio_video": relatorio_video,
            "relatorio_audio": relatorio_audio,
            "trilha": [],
        }
        if self.app is not None:
            return self.app.invoke(entrada)
        return self._executar_puro(entrada)
