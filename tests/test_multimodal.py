"""
Testes do módulo multimodal (Fase 4): vídeo (YOLOv8), áudio (triagem) e fusão.
Rodam offline: o áudio usa transcript mock e o vídeo usa o dataset/vídeo demo.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.multimodal.audio.analyze import analisar
from src.multimodal.audio.report import gerar_relatorio as rel_audio
from src.multimodal.audio.transcribe import Transcricao
from src.multimodal.fusion.multimodal_graph import FusaoMultimodal


# --------------------------------------------------------------------------- #
# Áudio — análise por léxico
# --------------------------------------------------------------------------- #
def test_analise_detecta_violencia_como_vermelho():
    texto = "Tenho medo dele, ele grita comigo e me empurrou. Não posso sair."
    a = analisar(texto, usar_llm=False)
    assert "violencia_domestica" in a.categorias()
    assert a.nivel_alerta == "vermelho"


def test_analise_registra_termos_para_explainability():
    a = analisar("Estou muito ansiosa e com medo.", usar_llm=False)
    assert a.sinais
    assert all(s.termos for s in a.sinais)  # todo sinal cita os termos que o dispararam


def test_analise_sem_sinais_fica_verde():
    a = analisar("A consulta correu bem, sem queixas.", usar_llm=False)
    assert a.nivel_alerta == "verde"


def test_relatorio_audio_estruturado():
    t = Transcricao(texto="Me sinto exausta, sem energia e sem vontade.", backend="mock")
    rel = rel_audio(t, analisar(t.texto, usar_llm=False))
    d = rel.to_dict()
    assert "nivel_alerta" in d and "sinais" in d


# --------------------------------------------------------------------------- #
# Fusão multimodal
# --------------------------------------------------------------------------- #
def _rel_video(nivel, anomalias=None):
    return {"nivel_alerta": nivel, "anomalias": anomalias or [], "instrumentos": []}


def _rel_audio(nivel, sinais=None):
    return {"nivel_alerta": nivel, "sinais": sinais or []}


def test_fusao_combina_pelo_maior_nivel():
    f = FusaoMultimodal(auditor=None)
    est = f.executar(_rel_video("verde"), _rel_audio("vermelho", ["possível violência doméstica"]),
                     paciente_id="PAC-0001")
    assert est["nivel_final"] == "vermelho"
    assert est.get("alerta")
    assert "consolidar" in est["trilha"]


def test_fusao_verde_nao_emite_alerta():
    f = FusaoMultimodal(auditor=None)
    est = f.executar(_rel_video("verde"), _rel_audio("verde"), paciente_id="PAC-0002")
    assert est["nivel_final"] == "verde"
    assert not est.get("alerta")
    assert "registrar_ok" in est["trilha"]


def test_fusao_agrega_achados_das_duas_modalidades():
    f = FusaoMultimodal(auditor=None)
    est = f.executar(
        _rel_video("amarelo", ["instrumento não previsto"]),
        _rel_audio("amarelo", ["ansiedade gestacional"]),
        paciente_id="PAC-0003",
    )
    txt = " ".join(est["achados"])
    assert "vídeo" in txt and "áudio" in txt


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
