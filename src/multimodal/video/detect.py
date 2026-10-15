"""
detect.py
---------
Inferência do detector de instrumentos (YOLOv8) em imagens e vídeos clínicos.

Carrega o `best.pt` treinado (se existir) ou cai no modelo base pré-treinado.
Devolve detecções ESTRUTURADAS (classe, confiança, bbox, tempo) — insumo para o
relatório especializado e para a camada de fusão multimodal. Também salva o
vídeo/imagem anotado para a demonstração.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .. import config


@dataclass
class Deteccao:
    classe: str
    confianca: float
    bbox: List[float]          # [x1, y1, x2, y2] em pixels
    frame: int = 0
    tempo_s: float = 0.0


@dataclass
class ResultadoVideo:
    """Resumo estruturado da análise de um vídeo."""
    fonte: str
    n_frames: int
    duracao_s: float
    deteccoes: List[Deteccao] = field(default_factory=list)
    contagem_por_classe: Dict[str, int] = field(default_factory=dict)
    video_anotado: Optional[str] = None
    modelo: str = ""

    def instrumentos_detectados(self) -> List[str]:
        return sorted(self.contagem_por_classe.keys())


def _carregar_modelo():
    from ultralytics import YOLO
    pesos = config.YOLO_WEIGHTS if config.YOLO_WEIGHTS.exists() else config.YOLO_BASE_MODEL
    return YOLO(str(pesos)), str(pesos)


def detectar_imagem(caminho: str, conf: float = 0.25) -> List[Deteccao]:
    """Roda a detecção em uma imagem e devolve a lista de detecções."""
    modelo, _ = _carregar_modelo()
    res = modelo.predict(caminho, conf=conf, verbose=False)[0]
    nomes = res.names
    saidas: List[Deteccao] = []
    for b in res.boxes:
        cls = nomes[int(b.cls)]
        xyxy = [round(float(v), 1) for v in b.xyxy[0].tolist()]
        saidas.append(Deteccao(classe=cls, confianca=round(float(b.conf), 3), bbox=xyxy))
    return saidas


def detectar_video(
    caminho: str,
    conf: float = 0.25,
    salvar_anotado: bool = True,
    passo_frames: int = 1,
) -> ResultadoVideo:
    """
    Roda a detecção quadro a quadro. Salva um vídeo anotado e devolve o resumo
    estruturado (detecções, contagem por classe, timeline).
    """
    import cv2

    modelo, pesos = _carregar_modelo()
    config.ensure_dirs()

    cap = cv2.VideoCapture(caminho)
    if not cap.isOpened():
        raise FileNotFoundError(f"não abriu o vídeo: {caminho}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    saida_path = None
    if salvar_anotado:
        saida_path = config.VIDEO_RESULTS_DIR / (Path(caminho).stem + "_anotado.mp4")
        writer = cv2.VideoWriter(
            str(saida_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (largura, altura)
        )

    deteccoes: List[Deteccao] = []
    contagem: Dict[str, int] = {}
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % passo_frames == 0:
            res = modelo.predict(frame, conf=conf, verbose=False)[0]
            nomes = res.names
            for b in res.boxes:
                cls = nomes[int(b.cls)]
                xyxy = [round(float(v), 1) for v in b.xyxy[0].tolist()]
                deteccoes.append(Deteccao(
                    classe=cls, confianca=round(float(b.conf), 3), bbox=xyxy,
                    frame=frame_idx, tempo_s=round(frame_idx / fps, 2),
                ))
                contagem[cls] = contagem.get(cls, 0) + 1
            if writer is not None:
                writer.write(res.plot())
        elif writer is not None:
            writer.write(frame)
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    return ResultadoVideo(
        fonte=caminho,
        n_frames=frame_idx,
        duracao_s=round(frame_idx / fps, 2),
        deteccoes=deteccoes,
        contagem_por_classe=contagem,
        video_anotado=str(saida_path) if saida_path else None,
        modelo=pesos,
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Detecção de instrumentos em vídeo/imagem.")
    ap.add_argument("caminho")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    if args.caminho.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        r = detectar_video(args.caminho, conf=args.conf)
        print(f"[ok] {r.n_frames} frames, {r.duracao_s}s")
        print(f"     instrumentos: {r.instrumentos_detectados()}")
        print(f"     contagem: {r.contagem_por_classe}")
        print(f"     vídeo anotado: {r.video_anotado}")
    else:
        dets = detectar_imagem(args.caminho, conf=args.conf)
        for d in dets:
            print(f"  {d.classe} ({d.confianca}) bbox={d.bbox}")
