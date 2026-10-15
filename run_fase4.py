"""
run_fase4.py
------------
Orquestra o fluxo multimodal da Fase 4 (saúde da mulher) de ponta a ponta:

  1. (opcional) treina o YOLOv8 no dataset sintético de demonstração;
  2. analisa um vídeo cirúrgico (detecção de instrumentos + relatório);
  3. transcreve e analisa um áudio de consulta (sinais psicológicos);
  4. funde as modalidades no grafo LangGraph e emite o alerta consolidado.

Exemplos:
    # tudo offline, usando os dados de demonstração incluídos
    python run_fase4.py

    # treina o detector demo antes (gera results/multimodal/video/best.pt)
    python run_fase4.py --treinar

    # aponta vídeo/áudio próprios
    python run_fase4.py --video caminho/cirurgia.mp4 --audio caminho/consulta.wav

Backends reais (quando disponíveis):
    YOLO treinado (best.pt)  ·  AUDIO_BACKEND=azure (F0) ou whisper  ·  LLM_BACKEND=ollama
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.multimodal import config
from src.multimodal.video.detect import detectar_video
from src.multimodal.video.report import gerar_relatorio as rel_video, formatar_texto as txt_video
from src.multimodal.audio.transcribe import transcrever
from src.multimodal.audio.analyze import analisar
from src.multimodal.audio.report import gerar_relatorio as rel_audio, formatar_texto as txt_audio
from src.multimodal.fusion.multimodal_graph import FusaoMultimodal

_VIDEO_DEMO = config.VIDEO_DATA_DIR / "demo_cirurgia.mp4"
_AUDIO_DEMO = config.AUDIO_DATA_DIR / "consulta_triagem_violencia.wav"


def main() -> None:
    ap = argparse.ArgumentParser(description="Fluxo multimodal da Fase 4.")
    ap.add_argument("--treinar", action="store_true", help="Treina o YOLOv8 demo antes.")
    ap.add_argument("--video", default=str(_VIDEO_DEMO))
    ap.add_argument("--audio", default=str(_AUDIO_DEMO))
    ap.add_argument("--paciente", default="PAC-0007")
    ap.add_argument("--procedimento", default="cirurgia ginecológica (demo)")
    args = ap.parse_args()

    if args.treinar:
        print("\n=== 0/4 Treino do detector (demo) ===")
        from src.multimodal.video import train_yolo
        train_yolo.run(mode="demo", epochs=12)

    print("\n=== 1/4 Análise de vídeo (YOLOv8) ===")
    rv = rel_video(detectar_video(args.video), procedimento=args.procedimento)
    print(txt_video(rv))

    print("\n=== 2/4 Análise de áudio (transcrição + triagem) ===")
    t = transcrever(args.audio)
    ra = rel_audio(t, analisar(t.texto))
    print(txt_audio(ra))

    print("\n=== 3/4 Fusão multimodal (LangGraph) ===")
    fusao = FusaoMultimodal()
    estado = fusao.executar(rv.to_dict(), ra.to_dict(), paciente_id=args.paciente)

    print("\n=== 4/4 Resultado consolidado ===")
    print(estado["resumo_final"])
    print("\n[ok] fluxo multimodal concluído.")


if __name__ == "__main__":
    main()
