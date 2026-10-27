"""Gera notebooks/09_multimodal_fase4.ipynb (demonstração da Fase 4)."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md("""# Fase 4 — Monitoramento Multimodal em Saúde da Mulher

**FIAP Pós-Tech — Machine Learning Engineering**

Demonstra o pipeline multimodal (vídeo + áudio + fusão):

1. Detecção de instrumentos cirúrgicos em vídeo (YOLOv8)
2. Transcrição + triagem psicológica de áudio (Azure/Whisper/mock)
3. Fusão em LangGraph → risco combinado → alerta à equipe
4. Segurança, explainability e auditoria (herdadas da Fase 3)

> Roda offline (backend mock/demo). Com Azure (F0) e um `best.pt` treinado no
> Colab, os mesmos passos usam os backends reais.""")

code("""import os, sys, json
sys.path.insert(0, os.path.abspath(".."))
os.environ.setdefault("LLM_BACKEND", "mock")
os.environ.setdefault("AUDIO_BACKEND", "auto")  # cai em mock offline""")

md("## 1. Vídeo — treino demo do YOLOv8 (offline)")
code("""from src.multimodal.video import train_yolo
out = train_yolo.run(mode="demo", epochs=8)   # gera dataset sintético + treina
print("pesos:", out["weights"])
print("mAP50:", out.get("mAP50"), "| mAP50-95:", out.get("mAP50_95"))""")

md("### 1.1 Inferência + relatório especializado")
code("""from src.multimodal.video.detect import detectar_video
from src.multimodal.video.report import gerar_relatorio, formatar_texto
res = detectar_video("../data/multimodal/video/demo_cirurgia.mp4")
rel_v = gerar_relatorio(res, procedimento="histerectomia (demo)")
print(formatar_texto(rel_v))
print("\\nVídeo anotado salvo em:", res.video_anotado)""")

md("""## 2. Áudio — transcrição + triagem psicológica

Cenário de triagem de violência doméstica (deve elevar o alerta a VERMELHO).""")
code("""from src.multimodal.audio.transcribe import transcrever
from src.multimodal.audio.analyze import analisar
from src.multimodal.audio.report import gerar_relatorio as rel_audio, formatar_texto as txt_audio
t = transcrever("../data/multimodal/audio/consulta_triagem_violencia.wav")
print("backend de transcrição:", t.backend)
rel_a = rel_audio(t, analisar(t.texto))
print(txt_audio(rel_a))""")

md("""## 3. Fusão multimodal (LangGraph)

Combina os relatórios de vídeo e áudio, decide o risco e emite o alerta.""")
code("""from src.multimodal.fusion.multimodal_graph import FusaoMultimodal
fusao = FusaoMultimodal()
estado = fusao.executar(rel_v.to_dict(), rel_a.to_dict(), paciente_id="PAC-0007")
print("LangGraph usado:", fusao.app is not None)
print()
print(estado["resumo_final"])""")

md("## 4. Auditoria (rastreabilidade da decisão)")
code("""from src.assistant.safety.audit_log import AuditLogger
ev = AuditLogger().ler_eventos()
print(f"{len(ev)} eventos auditados." if ev else "sem eventos ainda")
if ev:
    print(json.dumps(ev[-1], ensure_ascii=False, indent=2))""")

md("""---
### Conclusão

O pipeline funde vídeo e áudio num fluxo de decisão LangGraph que herda os
guardrails e a auditoria da Fase 3. Trocando os backends (YOLO real, Azure F0,
Ollama), a mesma solução opera em produção sem alterar a fusão.""")

nb["cells"] = cells
out = ROOT / "notebooks" / "09_multimodal_fase4.ipynb"
nbf.write(nb, str(out))
print("[ok]", out)
