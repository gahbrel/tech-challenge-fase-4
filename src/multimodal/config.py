"""
config.py
---------
Configuração central do módulo multimodal (Fase 4 — saúde da mulher).

Centraliza caminhos e flags para os módulos de vídeo (YOLOv8), áudio
(Azure Speech / Whisper) e fusão (LangGraph). Segue a mesma filosofia das fases
anteriores: roda em qualquer máquina, com o caminho "real" (GPU/nuvem) e um
caminho de demonstração/fallback offline.
"""

from __future__ import annotations

import os
from pathlib import Path

# Raiz do projeto (…/tech-challenge). Três níveis acima deste arquivo.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Dados de entrada -------------------------------------------------------- #
MM_DATA_DIR = PROJECT_ROOT / "data" / "multimodal"
VIDEO_DATA_DIR = MM_DATA_DIR / "video"
AUDIO_DATA_DIR = MM_DATA_DIR / "audio"

# Dataset sintético de demonstração para treinar/validar o YOLOv8 offline.
VIDEO_DEMO_DATASET = VIDEO_DATA_DIR / "demo_instrumentos"

# --- Saídas ----------------------------------------------------------------- #
RESULTS_DIR = PROJECT_ROOT / "results" / "multimodal"
VIDEO_RESULTS_DIR = RESULTS_DIR / "video"
AUDIO_RESULTS_DIR = RESULTS_DIR / "audio"
REPORTS_DIR = RESULTS_DIR / "reports"

# Pesos do YOLOv8 treinado (instrumentos). Se não existir, cai no pré-treinado.
YOLO_WEIGHTS = VIDEO_RESULTS_DIR / "best.pt"
YOLO_BASE_MODEL = os.environ.get("YOLO_BASE_MODEL", "yolov8n.pt")

# --- Áudio (Azure Speech F0 + fallback Whisper) ----------------------------- #
# AUDIO_BACKEND: "azure" | "whisper" | "auto" (auto → azure se houver chave, senão whisper)
AUDIO_BACKEND = os.environ.get("AUDIO_BACKEND", "auto")
AZURE_SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")

# --- Azure AI Language (Text Analytics): sentimento + termos-chave ----------- #
# Segundo serviço de nuvem (além do Speech). Fallback: léxico local.
AZURE_LANGUAGE_KEY = os.environ.get("AZURE_LANGUAGE_KEY", "")
AZURE_LANGUAGE_ENDPOINT = os.environ.get("AZURE_LANGUAGE_ENDPOINT", "")

# --- Sinais vitais (detecção de anomalias) ---------------------------------- #
VITALS_DATA_DIR = MM_DATA_DIR / "vitals"
VITALS_RESULTS_DIR = RESULTS_DIR / "vitals"
# VITALS_METHOD: "isolation_forest" | "zscore"
VITALS_METHOD = os.environ.get("VITALS_METHOD", "isolation_forest")

# --- Análise postural (pose) ------------------------------------------------ #
# YOLOv8-pose como alternativa prática e instalável ao OpenPose (mesmo fim:
# estimativa de pose 2D). Baixa o modelo dos releases do Ultralytics.
POSE_MODEL = os.environ.get("POSE_MODEL", "yolov8n-pose.pt")
POSE_RESULTS_DIR = RESULTS_DIR / "pose"

# --- Prescrições (detecção de anomalias na evolução) ------------------------ #
PRESCRIPTIONS_DATA_DIR = MM_DATA_DIR / "prescriptions"
PRESCRIPTIONS_RESULTS_DIR = RESULTS_DIR / "prescriptions"
# Variação relativa de dose considerada "brusca" (ex.: 0.5 = ±50%).
PRESC_VARIACAO_BRUSCA = float(os.environ.get("PRESC_VARIACAO_BRUSCA", "0.5"))

# Classes de instrumentos do dataset de demonstração (proxy dos instrumentos
# cirúrgicos ginecológicos do dataset público real usado no treino em GPU).
DEMO_CLASSES = ["bisturi", "pinca", "tesoura", "afastador"]


def ensure_dirs() -> None:
    for d in (
        VIDEO_DATA_DIR, AUDIO_DATA_DIR, VIDEO_RESULTS_DIR,
        AUDIO_RESULTS_DIR, REPORTS_DIR, VITALS_DATA_DIR,
        VITALS_RESULTS_DIR, POSE_RESULTS_DIR,
        PRESCRIPTIONS_DATA_DIR, PRESCRIPTIONS_RESULTS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
