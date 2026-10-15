"""
transcribe.py
-------------
Transcrição de áudio de consultas (Fase 4) — Speech-to-Text.

Backends intercambiáveis, na mesma filosofia do cliente LLM do projeto:

  - "azure"   : Azure AI Speech (tier gratuito F0). Caminho de nuvem exigido no
                enunciado. Requer AZURE_SPEECH_KEY e AZURE_SPEECH_REGION.
  - "whisper" : Whisper local (offline, sem nuvem). Requer o pacote `whisper`.
  - "mock"    : lê um transcript ao lado do áudio (`<audio>.txt`) ou devolve um
                texto-exemplo. Roda em qualquer máquina, sem rede nem modelo.
  - "auto"    : azure (se houver chave) → whisper (se instalado) → mock.

Assim a demonstração roda offline (mock/whisper) e, com a chave, usa o Azure de
verdade — marcando o requisito de integração com serviços em nuvem.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .. import config


@dataclass
class Transcricao:
    texto: str
    backend: str
    duracao_s: float = 0.0
    idioma: str = "pt-BR"
    fonte: str = ""


# --------------------------------------------------------------------------- #
# Resolução de backend
# --------------------------------------------------------------------------- #
def _resolver_backend(backend: str) -> str:
    if backend != "auto":
        return backend
    if config.AZURE_SPEECH_KEY and config.AZURE_SPEECH_REGION:
        return "azure"
    try:
        import whisper  # noqa: F401
        return "whisper"
    except Exception:
        return "mock"


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
def _transcrever_azure(caminho: str, idioma: str) -> str:
    """Azure AI Speech (F0). Código de produção; requer chave/região."""
    import azure.cognitiveservices.speech as speechsdk

    speech_config = speechsdk.SpeechConfig(
        subscription=config.AZURE_SPEECH_KEY, region=config.AZURE_SPEECH_REGION
    )
    speech_config.speech_recognition_language = idioma
    audio_config = speechsdk.audio.AudioConfig(filename=caminho)
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )

    # Reconhecimento contínuo (áudios maiores que uma única utterance).
    partes = []
    done = False

    def _parou(evt):
        nonlocal done
        done = True

    recognizer.recognized.connect(lambda e: partes.append(e.result.text))
    recognizer.session_stopped.connect(_parou)
    recognizer.canceled.connect(_parou)
    recognizer.start_continuous_recognition()
    import time
    while not done:
        time.sleep(0.3)
    recognizer.stop_continuous_recognition()
    return " ".join(p for p in partes if p).strip()


def _transcrever_whisper(caminho: str, idioma: str) -> str:
    """Whisper local (offline)."""
    import whisper
    modelo = whisper.load_model(config.WHISPER_MODEL)
    lang = idioma.split("-")[0]
    res = modelo.transcribe(caminho, language=lang)
    return (res.get("text") or "").strip()


def _transcrever_mock(caminho: str) -> str:
    """Lê o transcript ao lado do áudio (<audio>.txt) ou devolve um exemplo."""
    sidecar = Path(caminho).with_suffix(".txt")
    if sidecar.exists():
        return sidecar.read_text(encoding="utf-8").strip()
    return (
        "Não tenho dormido direito desde o parto. Me sinto muito cansada e sem "
        "vontade de nada, às vezes fico ansiosa e com medo sem saber explicar."
    )


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def transcrever(caminho: str, backend: Optional[str] = None, idioma: str = "pt-BR") -> Transcricao:
    resolvido = _resolver_backend(backend or config.AUDIO_BACKEND)
    if resolvido == "azure":
        try:
            texto = _transcrever_azure(caminho, idioma)
        except Exception:
            resolvido, texto = "mock", _transcrever_mock(caminho)
    elif resolvido == "whisper":
        try:
            texto = _transcrever_whisper(caminho, idioma)
        except Exception:
            resolvido, texto = "mock", _transcrever_mock(caminho)
    else:
        resolvido, texto = "mock", _transcrever_mock(caminho)

    dur = 0.0
    try:
        import wave
        with wave.open(caminho, "rb") as w:
            dur = round(w.getnframes() / float(w.getframerate()), 2)
    except Exception:
        pass

    return Transcricao(texto=texto, backend=resolvido, duracao_s=dur,
                       idioma=idioma, fonte=caminho)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Transcrição de áudio de consulta.")
    ap.add_argument("caminho")
    ap.add_argument("--backend", default=None, choices=[None, "azure", "whisper", "mock", "auto"])
    args = ap.parse_args()
    t = transcrever(args.caminho, backend=args.backend)
    print(f"[backend: {t.backend}] ({t.duracao_s}s)")
    print(t.texto)
