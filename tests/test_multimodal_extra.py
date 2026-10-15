"""
Testes das extensões da Fase 4: sinais vitais, acústica, postura, Text Analytics
(fallback) e a fusão com 4 modalidades. Rodam offline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.multimodal.vitals import detector as vitals
from src.multimodal.video import pose
from src.multimodal.audio.analyze import analisar
from src.multimodal.fusion.multimodal_graph import FusaoMultimodal


# --------------------------------------------------------------------------- #
# Sinais vitais
# --------------------------------------------------------------------------- #
def test_vitais_detecta_preeclampsia_e_bradicardia():
    csv_path = vitals.gerar_serie_demo(paciente_id="PAC-T", seed=7)
    rel = vitals.analisar(str(csv_path), paciente_id="PAC-T")
    assert rel.nivel_alerta == "vermelho"
    motivos = " ".join(a.motivo for a in rel.anomalias)
    assert "pré-eclâmpsia" in motivos
    assert any("bradicardia" in a.motivo for a in rel.anomalias)


def test_vitais_sem_anomalias_fica_verde():
    csv_path = vitals.gerar_serie_demo(paciente_id="PAC-OK", seed=1, com_anomalias=False)
    rel = vitals.analisar(str(csv_path), paciente_id="PAC-OK", metodo="zscore")
    assert rel.nivel_alerta in ("verde", "amarelo")  # z-score pode pegar ruído leve
    assert not any(a.gravidade == "vermelho" for a in rel.anomalias)


def test_vitais_metodo_zscore_funciona():
    csv_path = vitals.gerar_serie_demo(paciente_id="PAC-Z", seed=7)
    rel = vitals.analisar(str(csv_path), paciente_id="PAC-Z", metodo="zscore")
    assert rel.metodo == "zscore"
    assert rel.n_amostras == 60


# --------------------------------------------------------------------------- #
# Postura (pose)
# --------------------------------------------------------------------------- #
def test_pose_desconforto_gera_sinais():
    rel = pose.analisar_keypoints(pose.gerar_keypoints_demo("desconforto"),
                                  fonte="t", modelo="demo")
    assert rel.nivel_alerta == "amarelo"
    assert rel.sinais


def test_pose_normal_fica_verde():
    rel = pose.analisar_keypoints(pose.gerar_keypoints_demo("normal"),
                                  fonte="t", modelo="demo")
    assert rel.nivel_alerta == "verde"


def test_pose_vazio_nao_quebra():
    import numpy as np
    rel = pose.analisar_keypoints(np.empty((0, 17, 2)), fonte="t")
    assert rel.n_frames == 0


# --------------------------------------------------------------------------- #
# Áudio: Text Analytics (fallback léxico) + acústica
# --------------------------------------------------------------------------- #
def test_text_analytics_fallback_lexico():
    a = analisar("Estou muito ansiosa e com medo.", usar_llm=False)
    assert a.backend_texto == "lexico"
    assert a.sentimento in ("negative", "neutral")
    assert a.termos_chave  # deriva dos termos do léxico


def test_acustica_opcional_nao_quebra_sem_audio():
    a = analisar("consulta tranquila, sem queixas", usar_llm=False, caminho_audio=None)
    assert a.acustica == {}


# --------------------------------------------------------------------------- #
# Fusão com 4 modalidades
# --------------------------------------------------------------------------- #
def test_fusao_quatro_modalidades_pega_maior_nivel():
    f = FusaoMultimodal(auditor=None)
    est = f.executar(
        {"nivel_alerta": "verde", "anomalias": []},
        {"nivel_alerta": "amarelo", "sinais": ["ansiedade gestacional"]},
        paciente_id="PAC-1",
        relatorio_vitais={"nivel_alerta": "vermelho", "sinais": ["pressao_arterial (pré-eclâmpsia)"]},
        relatorio_postura={"nivel_alerta": "amarelo", "sinais": ["assimetria postural dos ombros"]},
    )
    assert est["nivel_final"] == "vermelho"
    txt = " ".join(est["achados"])
    assert "sinais vitais" in txt and "postura" in txt and "áudio" in txt
    assert "receber_vitais" in est["trilha"] and "receber_postura" in est["trilha"]


def test_fusao_retrocompativel_dois_argumentos():
    # chamada antiga (só vídeo+áudio) deve continuar funcionando
    f = FusaoMultimodal(auditor=None)
    est = f.executar({"nivel_alerta": "verde", "anomalias": []},
                     {"nivel_alerta": "verde", "sinais": []}, paciente_id="PAC-2")
    assert est["nivel_final"] == "verde"
    assert "registrar_ok" in est["trilha"]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
