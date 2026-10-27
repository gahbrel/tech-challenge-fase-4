"""
Testes do módulo de anomalias na evolução de prescrições (Fase 4) e sua
integração na fusão (5 modalidades). Rodam offline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.multimodal.prescriptions import detector as presc
from src.multimodal.fusion.multimodal_graph import FusaoMultimodal


def test_detecta_dose_brusca_troca_e_suspensao():
    rel = presc.analisar(str(presc.gerar_historico_demo("PAC-P")), "PAC-P")
    tipos = {a.tipo for a in rel.anomalias}
    assert "dose_brusca" in tipos
    assert "troca" in tipos
    assert "suspensao" in tipos
    assert rel.nivel_alerta == "vermelho"


def test_sem_anomalias_fica_verde():
    csv = presc.gerar_historico_demo("PAC-OK", com_anomalias=False)
    rel = presc.analisar(str(csv), "PAC-OK")
    assert rel.nivel_alerta == "verde"
    assert not rel.anomalias


def test_dose_brusca_respeita_limiar():
    # com limiar alto (300%), o aumento de +200% não conta como brusco
    rel = presc.analisar(str(presc.gerar_historico_demo("PAC-L")), "PAC-L",
                         variacao_brusca=3.0)
    assert not any(a.tipo == "dose_brusca" for a in rel.anomalias)


def test_relatorio_estruturado_para_fusao():
    rel = presc.analisar(str(presc.gerar_historico_demo("PAC-D")), "PAC-D")
    d = rel.to_dict()
    assert "nivel_alerta" in d and "sinais" in d
    assert all(isinstance(s, str) for s in d["sinais"])


def test_fusao_inclui_prescricoes():
    f = FusaoMultimodal(auditor=None)
    est = f.executar(
        {"nivel_alerta": "verde", "anomalias": []},
        {"nivel_alerta": "verde", "sinais": []},
        paciente_id="PAC-F",
        relatorio_vitais={"nivel_alerta": "verde", "sinais": []},
        relatorio_postura={"nivel_alerta": "verde", "sinais": []},
        relatorio_prescricoes={"nivel_alerta": "vermelho",
                               "sinais": ["metildopa: suspensão sem desmame"]},
    )
    assert est["nivel_final"] == "vermelho"
    assert "receber_prescricoes" in est["trilha"]
    assert any("prescrições" in a for a in est["achados"])


def test_fusao_retrocompativel_sem_prescricoes():
    # chamada sem prescrições continua válida
    f = FusaoMultimodal(auditor=None)
    est = f.executar({"nivel_alerta": "verde", "anomalias": []},
                     {"nivel_alerta": "verde", "sinais": []}, paciente_id="PAC-R")
    assert est["nivel_final"] == "verde"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
