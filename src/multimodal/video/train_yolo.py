"""
train_yolo.py
-------------
Treinamento do detector de instrumentos cirúrgicos ginecológicos (YOLOv8).

Dois caminhos, na mesma filosofia demo/real das fases anteriores:

  --mode real : treina o YOLOv8 sobre um dataset público em formato YOLO
                (ex.: um dataset de "surgical instruments" do Roboflow Universe,
                ou o m2cai16-tool). Requer GPU e o `data.yaml` do dataset. É o
                caminho de produção — recomendado rodar no Google Colab (GPU
                grátis), exportar o `best.pt` e versioná-lo para inferência em CPU.

  --mode demo : (padrão) gera um dataset SINTÉTICO de instrumentos (formas
                geométricas rotuladas como bisturi/pinça/tesoura/afastador) e
                treina o YOLOv8n por poucas épocas. Roda em qualquer máquina, sem
                GPU nem download, e produz um `best.pt` REAL — comprovando o
                pipeline de treino ponta a ponta.

Como baixar o dataset real (no Colab), exemplo com Roboflow:

    from roboflow import Roboflow
    rf = Roboflow(api_key="SUA_CHAVE")
    ds = rf.workspace("<ws>").project("<surgical-instruments>").version(1).download("yolov8")
    # depois: python -m src.multimodal.video.train_yolo --mode real --data {ds.location}/data.yaml

Saídas: results/multimodal/video/best.pt + métricas do treino.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Optional, Tuple

from .. import config


# --------------------------------------------------------------------------- #
# Dataset sintético de demonstração (roda offline)
# --------------------------------------------------------------------------- #
def _desenha_instrumento(img, classe: int, w: int, h: int, rng: random.Random) -> Tuple[float, float, float, float]:
    """Desenha uma forma que representa um instrumento e devolve a bbox YOLO."""
    import cv2

    cor = (200, 200, 210)
    cx, cy = rng.randint(int(w * 0.25), int(w * 0.75)), rng.randint(int(h * 0.25), int(h * 0.75))
    bw, bh = rng.randint(int(w * 0.12), int(w * 0.28)), rng.randint(int(h * 0.12), int(h * 0.28))
    x1, y1, x2, y2 = cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2

    if classe == 0:      # bisturi — retângulo fino (lâmina)
        cv2.rectangle(img, (x1, cy - 4), (x2, cy + 4), cor, -1)
        cv2.rectangle(img, (x1, y1), (x1 + bw // 4, y2), (120, 120, 130), -1)
    elif classe == 1:    # pinça — duas linhas convergentes
        cv2.line(img, (x1, y1), (x2, cy), cor, 4)
        cv2.line(img, (x1, y2), (x2, cy), cor, 4)
    elif classe == 2:    # tesoura — X + anéis
        cv2.line(img, (x1, y1), (x2, y2), cor, 4)
        cv2.line(img, (x1, y2), (x2, y1), cor, 4)
        cv2.circle(img, (x1, y1), 6, cor, 2)
        cv2.circle(img, (x1, y2), 6, cor, 2)
    else:                # afastador — U
        cv2.line(img, (x1, y1), (x1, y2), cor, 4)
        cv2.line(img, (x1, y2), (x2, y2), cor, 4)
        cv2.line(img, (x2, y2), (x2, y1), cor, 4)

    # bbox normalizada (YOLO)
    return (cx / w, cy / h, bw / w, bh / h)


def gerar_dataset_demo(n_train: int = 48, n_val: int = 12, seed: int = 42) -> Path:
    """Gera um dataset YOLO sintético e o data.yaml. Retorna o caminho do yaml."""
    import cv2
    import numpy as np

    rng = random.Random(seed)
    base = config.VIDEO_DEMO_DATASET
    for split, n in (("train", n_train), ("val", n_val)):
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            w, h = 416, 416
            img = np.full((h, w, 3), rng.randint(25, 45), np.uint8)  # fundo "tecido"
            linhas = []
            for _ in range(rng.randint(1, 3)):
                classe = rng.randint(0, len(config.DEMO_CLASSES) - 1)
                cx, cy, bw, bh = _desenha_instrumento(img, classe, w, h, rng)
                linhas.append(f"{classe} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            cv2.imwrite(str(base / "images" / split / f"img_{i:03d}.jpg"), img)
            (base / "labels" / split / f"img_{i:03d}.txt").write_text("\n".join(linhas))

    nomes = config.DEMO_CLASSES
    yaml_txt = (
        f"path: {base}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(nomes)}\n"
        f"names: {nomes}\n"
    )
    yaml_path = base / "data.yaml"
    yaml_path.write_text(yaml_txt)
    return yaml_path


# --------------------------------------------------------------------------- #
# Treino
# --------------------------------------------------------------------------- #
def treinar(data_yaml: Path, epochs: int, imgsz: int = 416, base_model: Optional[str] = None) -> dict:
    """Treina o YOLOv8 sobre `data_yaml` e copia o melhor peso para results/."""
    import shutil
    from ultralytics import YOLO

    config.ensure_dirs()
    modelo = YOLO(base_model or config.YOLO_BASE_MODEL)
    res = modelo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        project=str(config.VIDEO_RESULTS_DIR / "runs"),
        name="instrumentos",
        exist_ok=True,
        verbose=False,
        plots=False,
    )
    # localiza o best.pt e copia para o caminho padrão
    best = Path(res.save_dir) / "weights" / "best.pt"
    if best.exists():
        shutil.copy(best, config.YOLO_WEIGHTS)

    metricas = {}
    try:
        metricas = {
            "mAP50": round(float(res.box.map50), 4),
            "mAP50_95": round(float(res.box.map), 4),
        }
    except Exception:
        pass
    return {"weights": str(config.YOLO_WEIGHTS), "save_dir": str(res.save_dir), **metricas}


def run(mode: str = "demo", data: Optional[str] = None, epochs: int = 5) -> dict:
    if mode == "real":
        if not data:
            raise SystemExit("--mode real exige --data <caminho para data.yaml do dataset>")
        return treinar(Path(data), epochs=epochs)

    # demo
    print("[demo] gerando dataset sintético de instrumentos…")
    yaml_path = gerar_dataset_demo()
    print(f"[demo] treinando YOLOv8n por {epochs} épocas (offline)…")
    return treinar(yaml_path, epochs=epochs, imgsz=416)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Treino do YOLOv8 (instrumentos cirúrgicos).")
    ap.add_argument("--mode", choices=["demo", "real"], default="demo")
    ap.add_argument("--data", default=None, help="Caminho do data.yaml (modo real).")
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()

    out = run(mode=args.mode, data=args.data, epochs=args.epochs)
    print(f"[ok] treino concluído — pesos em {out['weights']}")
    if "mAP50" in out:
        print(f"     mAP50={out['mAP50']} | mAP50-95={out['mAP50_95']}")
