"""
acoustics.py
------------
Extração de características ACÚSTICAS da voz (Fase 4) — complementa a análise do
texto transcrito com o modo COMO a paciente fala (não só o que fala).

Usando `librosa`, extrai marcadores associados a fadiga/estresse/baixo ânimo:
  - energia (RMS médio) — voz apática/baixa energia
  - tom / pitch (F0 médio e variação) — monotonia ou tensão
  - ritmo de fala (taxa de onsets) — fala lenta
  - proporção de pausas (silêncio) — hesitação/desânimo

Aplica heurísticas simples e explicáveis (limiar por marcador) para sinalizar
"fadiga/baixo ânimo" e "tensão/estresse". É triagem de apoio, não diagnóstico.
Degrada com segurança: sem `librosa` ou com áudio sem sinal, retorna vazio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FeaturesAcusticas:
    energia_rms: float = 0.0
    pitch_medio_hz: float = 0.0
    pitch_variacao: float = 0.0
    taxa_fala: float = 0.0          # onsets por segundo
    proporcao_pausas: float = 0.0   # 0..1
    duracao_s: float = 0.0
    sinais: List[str] = field(default_factory=list)   # ex.: "fadiga/baixo ânimo"
    disponivel: bool = False

    def to_dict(self) -> Dict:
        return {
            "energia_rms": round(self.energia_rms, 4),
            "pitch_medio_hz": round(self.pitch_medio_hz, 1),
            "pitch_variacao": round(self.pitch_variacao, 1),
            "taxa_fala": round(self.taxa_fala, 2),
            "proporcao_pausas": round(self.proporcao_pausas, 2),
            "sinais_acusticos": self.sinais,
            "disponivel": self.disponivel,
        }


def extrair(caminho_audio: str) -> FeaturesAcusticas:
    """Extrai as features acústicas de um arquivo de áudio."""
    try:
        import librosa
        import numpy as np
    except Exception:
        return FeaturesAcusticas()

    try:
        y, sr = librosa.load(caminho_audio, sr=16000, mono=True)
    except Exception:
        return FeaturesAcusticas()

    if y is None or len(y) == 0 or float(np.max(np.abs(y))) < 1e-4:
        # áudio vazio/silencioso — sem sinal acústico utilizável
        return FeaturesAcusticas(duracao_s=round(len(y) / sr, 2) if sr else 0.0)

    dur = len(y) / sr
    rms = float(np.mean(librosa.feature.rms(y=y)))

    # Pitch (F0) via pyin, restrito à faixa da voz humana.
    try:
        f0, voiced, _ = librosa.pyin(y, fmin=70, fmax=400, sr=sr)
        f0v = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        pitch_medio = float(np.mean(f0v)) if f0v.size else 0.0
        pitch_var = float(np.std(f0v)) if f0v.size else 0.0
    except Exception:
        pitch_medio, pitch_var = 0.0, 0.0

    # Ritmo: taxa de onsets por segundo.
    try:
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
        taxa_fala = len(onsets) / dur if dur else 0.0
    except Exception:
        taxa_fala = 0.0

    # Pausas: proporção de frames com energia abaixo de um limiar relativo.
    try:
        rms_frames = librosa.feature.rms(y=y)[0]
        limiar = 0.3 * float(np.mean(rms_frames))
        prop_pausas = float(np.mean(rms_frames < limiar))
    except Exception:
        prop_pausas = 0.0

    feats = FeaturesAcusticas(
        energia_rms=rms, pitch_medio_hz=pitch_medio, pitch_variacao=pitch_var,
        taxa_fala=taxa_fala, proporcao_pausas=prop_pausas, duracao_s=round(dur, 2),
        disponivel=True,
    )

    # Heurísticas de triagem (limiares de demonstração).
    sinais: List[str] = []
    if rms < 0.02 and prop_pausas > 0.5:
        sinais.append("fadiga/baixo ânimo (voz de baixa energia e muitas pausas)")
    if taxa_fala and taxa_fala < 1.0:
        sinais.append("fala lenta/hesitante")
    if pitch_var > 60:
        sinais.append("tensão/estresse (grande variação de tom)")
    feats.sinais = sinais
    return feats
