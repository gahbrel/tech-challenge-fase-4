"""
pose.py
-------
Análise POSTURAL / de movimento em vídeo (Fase 4) — ex.: sessões de fisioterapia
pós-parto e sinais não-verbais de desconforto em consultas.

Usa **YOLOv8-pose** (Ultralytics) para estimar os keypoints do corpo. O YOLOv8-pose
é adotado como alternativa prática e instalável ao OpenPose citado no enunciado —
ambos fazem estimativa de pose 2D; o YOLOv8-pose roda sobre o mesmo pacote já
usado na detecção de instrumentos e dispensa a compilação do OpenPose.

A partir dos keypoints (padrão COCO, 17 pontos), calcula marcadores simples e
explicáveis:
  - movimento médio dos keypoints entre frames (pouco movimento → rigidez/
    desconforto; útil em fisioterapia);
  - assimetria postural (diferença de altura entre ombros e entre quadris);
  - postura defensiva (ombros encolhidos/elevados em relação ao pescoço).

Caminho real: `analisar_video()` sobre um vídeo com pessoa. Caminho de demo:
`gerar_keypoints_demo()` + `analisar_keypoints()` roda offline e mostra a lógica.
Triagem de apoio — não diagnostica.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .. import config

# Índices COCO usados.
_NARIZ, _OMBRO_E, _OMBRO_D, _QUADRIL_E, _QUADRIL_D = 0, 5, 6, 11, 12


@dataclass
class RelatorioPostura:
    fonte: str
    n_frames: int
    movimento_medio: float = 0.0
    assimetria_ombros: float = 0.0
    assimetria_quadris: float = 0.0
    presenca: float = 0.0            # fração de frames com pessoa detectada
    sinais: List[str] = field(default_factory=list)
    nivel_alerta: str = "verde"
    modelo: str = ""

    def to_dict(self) -> Dict:
        return {
            "fonte": self.fonte,
            "n_frames": self.n_frames,
            "movimento_medio": round(self.movimento_medio, 4),
            "assimetria_ombros": round(self.assimetria_ombros, 4),
            "assimetria_quadris": round(self.assimetria_quadris, 4),
            "presenca": round(self.presenca, 2),
            "sinais": self.sinais,
            "nivel_alerta": self.nivel_alerta,
            "modelo": self.modelo,
        }


def analisar_keypoints(seq, fonte: str = "keypoints", modelo: str = "demo") -> RelatorioPostura:
    """
    Recebe uma sequência de keypoints (lista/array de shape [frames, 17, 2],
    coordenadas normalizadas 0..1; NaN = ponto ausente) e devolve o relatório.
    """
    import numpy as np
    arr = np.asarray(seq, dtype=float)
    n = int(arr.shape[0]) if arr.ndim == 3 else 0
    if n == 0:
        return RelatorioPostura(fonte=fonte, n_frames=0, modelo=modelo)

    # presença: frames em que os pontos-chave do tronco estão válidos
    # (nariz, ombros e quadris) — robusto a extremidades ausentes.
    core = [_NARIZ, _OMBRO_E, _OMBRO_D, _QUADRIL_E, _QUADRIL_D]
    core_valido = ~np.isnan(arr[:, core, :]).any(axis=2)        # [frames, 5]
    presenca = float(np.mean(core_valido.mean(axis=1) >= 0.8))

    # movimento médio: deslocamento médio dos keypoints entre frames consecutivos
    difs = np.linalg.norm(np.diff(arr, axis=0), axis=2)   # [frames-1, 17]
    movimento = float(np.nanmean(difs)) if difs.size else 0.0

    # assimetria: diferença média de altura (y) entre lados
    def _assimetria(ie, idr):
        d = np.abs(arr[:, ie, 1] - arr[:, idr, 1])
        return float(np.nanmean(d))
    assim_ombros = _assimetria(_OMBRO_E, _OMBRO_D)
    assim_quadris = _assimetria(_QUADRIL_E, _QUADRIL_D)

    sinais: List[str] = []
    if presenca >= 0.3:
        if movimento < 0.004:
            sinais.append("movimento reduzido (possível rigidez/desconforto)")
        if assim_ombros > 0.05:
            sinais.append("assimetria postural dos ombros")
        if assim_quadris > 0.05:
            sinais.append("assimetria postural dos quadris")
        # postura defensiva: ombros próximos/acima da linha do nariz
        ombro_y = np.nanmean((arr[:, _OMBRO_E, 1] + arr[:, _OMBRO_D, 1]) / 2)
        nariz_y = np.nanmean(arr[:, _NARIZ, 1])
        if ombro_y - nariz_y < 0.12:
            sinais.append("postura defensiva (ombros encolhidos/elevados)")

    nivel = "amarelo" if sinais else "verde"
    return RelatorioPostura(
        fonte=fonte, n_frames=n, movimento_medio=movimento,
        assimetria_ombros=assim_ombros, assimetria_quadris=assim_quadris,
        presenca=presenca, sinais=sinais, nivel_alerta=nivel, modelo=modelo,
    )


def analisar_video(caminho: str, passo_frames: int = 3) -> RelatorioPostura:
    """Caminho REAL: estima pose com YOLOv8-pose e analisa os keypoints."""
    import numpy as np
    from ultralytics import YOLO

    modelo = YOLO(config.POSE_MODEL)
    res = modelo.predict(caminho, stream=True, verbose=False)
    seq = []
    for i, r in enumerate(res):
        if i % passo_frames:
            continue
        kp = r.keypoints
        if kp is None or kp.xyn is None or len(kp.xyn) == 0:
            seq.append(np.full((17, 2), np.nan))
        else:
            seq.append(kp.xyn[0].cpu().numpy())  # 1ª pessoa
    if not seq:
        return RelatorioPostura(fonte=caminho, n_frames=0, modelo=config.POSE_MODEL)
    return analisar_keypoints(np.array(seq), fonte=caminho, modelo=config.POSE_MODEL)


# --------------------------------------------------------------------------- #
# Demo offline: sequências de keypoints sintéticas
# --------------------------------------------------------------------------- #
def gerar_keypoints_demo(cenario: str = "desconforto", n: int = 40, seed: int = 3):
    """
    Gera uma sequência de keypoints sintética para demonstrar a análise offline.
    cenario: "normal" | "desconforto" (pouco movimento + assimetria).
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    base = np.full((17, 2), np.nan)
    # posições aproximadas (x, y) normalizadas de uma pessoa em pé, de frente
    base[_NARIZ] = [0.50, 0.20]
    base[_OMBRO_E] = [0.42, 0.35]
    base[_OMBRO_D] = [0.58, 0.35]
    base[_QUADRIL_E] = [0.45, 0.60]
    base[_QUADRIL_D] = [0.55, 0.60]

    seq = []
    for t in range(n):
        f = base.copy()
        if cenario == "desconforto":
            amp = 0.002                      # movimento mínimo
            f[_OMBRO_E, 1] += 0.07           # ombro esquerdo mais alto (assimetria)
            f[_OMBRO_E, 1] -= 0.0; f[_OMBRO_D, 1] += 0.0
            f[[_OMBRO_E, _OMBRO_D], 1] -= 0.10   # ombros encolhidos (defensiva)
        else:
            amp = 0.02                       # movimento normal
        ruido = rng.normal(0, amp, size=(17, 2))
        seq.append(f + ruido)
    return np.array(seq)


if __name__ == "__main__":
    kp = gerar_keypoints_demo("desconforto")
    rel = analisar_keypoints(kp, fonte="demo_fisioterapia", modelo="demo")
    import json
    print(json.dumps(rel.to_dict(), ensure_ascii=False, indent=2))
