"""
run_fase4.py
------------
Orquestra o fluxo multimodal da Fase 4 (saúde da mulher) de ponta a ponta:

  1. (opcional) treina o YOLOv8 no dataset sintético de demonstração;
  2. vídeo: detecção de instrumentos (YOLOv8) + análise postural (YOLOv8-pose);
  3. áudio: transcrição (Azure/Whisper) + triagem (léxico/Text Analytics) + acústica;
  4. sinais vitais: detecção de anomalias (Isolation Forest + regras clínicas);
  5. funde as modalidades no grafo LangGraph e emite o alerta consolidado.

Exemplos:
    # tudo offline, usando os dados de demonstração incluídos
    python run_fase4.py

    # treina o detector demo antes (gera results/multimodal/video/best.pt)
    python run_fase4.py --treinar

    # aponta vídeo/áudio próprios
    python run_fase4.py --video caminho/cirurgia.mp4 --audio caminho/consulta.wav

Backends reais (quando disponíveis):
    YOLO treinado (best.pt)  ·  AUDIO_BACKEND=azure (Speech F0)  ·  Azure Text
    Analytics (AZURE_LANGUAGE_KEY)  ·  LLM_BACKEND=ollama  ·  pose em vídeo real
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.multimodal import config
from src.multimodal.video.detect import detectar_video
from src.multimodal.video.report import gerar_relatorio as rel_video, formatar_texto as txt_video
from src.multimodal.video import pose
from src.multimodal.audio.transcribe import transcrever
from src.multimodal.audio.analyze import analisar
from src.multimodal.audio.report import gerar_relatorio as rel_audio, formatar_texto as txt_audio
from src.multimodal.vitals import detector as vitals
from src.multimodal.prescriptions import detector as prescricoes
from src.multimodal.fusion.multimodal_graph import FusaoMultimodal

_VIDEO_DEMO = config.VIDEO_DATA_DIR / "demo_cirurgia.mp4"
_AUDIO_DEMO = config.AUDIO_DATA_DIR / "consulta_triagem_violencia.wav"


def main() -> None:
    ap = argparse.ArgumentParser(description="Fluxo multimodal da Fase 4.")
    ap.add_argument("--treinar", action="store_true", help="Treina o YOLOv8 demo antes.")
    ap.add_argument("--video", default=str(_VIDEO_DEMO))
    ap.add_argument("--audio", default=str(_AUDIO_DEMO))
    ap.add_argument("--vitais", default=None, help="CSV de sinais vitais (senão, gera demo).")
    ap.add_argument("--prescricoes", default=None, help="CSV de prescrições (senão, gera demo).")
    ap.add_argument("--pose-video", default=None, help="Vídeo p/ análise postural real (senão, demo).")
    ap.add_argument("--paciente", default="PAC-0007")
    ap.add_argument("--procedimento", default="cirurgia ginecológica (demo)")
    args = ap.parse_args()

    if args.treinar:
        print("\n=== 0/5 Treino do detector (demo) ===")
        from src.multimodal.video import train_yolo
        train_yolo.run(mode="demo", epochs=12)

    print("\n=== 1/5 Vídeo: instrumentos (YOLOv8) + postura (YOLOv8-pose) ===")
    rv = rel_video(detectar_video(args.video), procedimento=args.procedimento)
    print(txt_video(rv))
    if args.pose_video:
        rp = pose.analisar_video(args.pose_video)
    else:
        # demo offline: sequência de keypoints sintética (cenário de desconforto)
        rp = pose.analisar_keypoints(pose.gerar_keypoints_demo("desconforto"),
                                     fonte="demo_fisioterapia", modelo="demo")
    print(f"[postura] nível={rp.nivel_alerta} | sinais={rp.sinais or 'nenhum'}")

    print("\n=== 2/5 Áudio: transcrição + triagem + acústica ===")
    t = transcrever(args.audio)
    ra = rel_audio(t, analisar(t.texto, caminho_audio=args.audio))
    print(txt_audio(ra))

    print("\n=== 3/6 Sinais vitais: detecção de anomalias ===")
    csv_path = args.vitais or str(vitals.gerar_serie_demo(paciente_id=args.paciente))
    rvit = vitals.analisar(csv_path, paciente_id=args.paciente)
    print(vitals.formatar_texto(rvit))

    print("\n=== 4/6 Prescrições: anomalias na evolução ===")
    presc_csv = args.prescricoes or str(prescricoes.gerar_historico_demo(paciente_id=args.paciente))
    rpresc = prescricoes.analisar(presc_csv, paciente_id=args.paciente)
    print(prescricoes.formatar_texto(rpresc))

    print("\n=== 5/6 Fusão multimodal (LangGraph) ===")
    fusao = FusaoMultimodal()
    estado = fusao.executar(
        rv.to_dict(), ra.to_dict(), paciente_id=args.paciente,
        relatorio_vitais=rvit.to_dict(), relatorio_postura=rp.to_dict(),
        relatorio_prescricoes=rpresc.to_dict(),
    )

    print("\n=== 6/6 Resultado consolidado ===")
    print(estado["resumo_final"])
    print("\n[ok] fluxo multimodal concluído.")


if __name__ == "__main__":
    main()
