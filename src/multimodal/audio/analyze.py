"""
analyze.py
----------
Análise de sinais de bem-estar psicológico feminino a partir do transcript
(Fase 4). Rastreia indícios de:

  - ansiedade gestacional
  - depressão pós-parto
  - fadiga hormonal
  - possível violência doméstica

Método primário: LÉXICO ponderado (determinístico, offline e explicável — cada
sinal registra os termos que o dispararam, para rastreabilidade). Opcionalmente,
enriquece com a LLM do projeto (Ollama/mock) para uma leitura em linguagem
natural. É uma TRIAGEM de apoio — não é diagnóstico. Toda saída leva o aviso de
validação humana (reaproveitando os guardrails da Fase 3).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Léxico por categoria (termos em minúsculas, sem acento — o texto é normalizado).
_LEXICO: Dict[str, List[str]] = {
    "ansiedade": [
        "ansiosa", "ansiedade", "nervosa", "preocupada", "medo", "aflita",
        "coracao acelerado", "sufoco", "panico", "tensa", "inquieta",
    ],
    "depressao_pos_parto": [
        "sem vontade", "vazia", "chorar", "choro", "cansada", "exausta",
        "sem energia", "culpa", "nao consigo cuidar", "tristeza", "triste",
        "sem dormir", "nao durmo", "desanimada", "sozinha",
    ],
    "fadiga_hormonal": [
        "cansada", "exausta", "sem energia", "calor", "ondas de calor",
        "insonia", "irritada", "oscilacao de humor",
    ],
    "violencia_domestica": [
        "ele me bateu", "me empurrou", "tenho medo dele", "me ameacou",
        "nao posso sair", "ele controla", "me machucou", "grita comigo",
        "com medo em casa", "me proibe",
    ],
}

# Rótulos legíveis.
_ROTULOS = {
    "ansiedade": "ansiedade gestacional",
    "depressao_pos_parto": "depressão pós-parto",
    "fadiga_hormonal": "fadiga hormonal",
    "violencia_domestica": "possível violência doméstica",
}

AVISO = (
    "Triagem de apoio à decisão (não é diagnóstico). Requer avaliação e "
    "validação de profissional de saúde. Em caso de risco, acionar o protocolo "
    "de proteção e a rede de apoio."
)


@dataclass
class SinalDetectado:
    categoria: str
    rotulo: str
    termos: List[str]
    intensidade: float          # 0..1 (proporcional aos termos encontrados)


@dataclass
class AnaliseAudio:
    transcript: str
    sinais: List[SinalDetectado] = field(default_factory=list)
    nivel_alerta: str = "verde"        # verde | amarelo | vermelho
    resumo_llm: str = ""
    backend_llm: str = ""
    # Azure Text Analytics (ou fallback léxico)
    sentimento: str = ""               # positive | neutral | negative | mixed
    termos_chave: List[str] = field(default_factory=list)
    backend_texto: str = ""            # "azure" | "lexico"
    # Características acústicas (librosa)
    acustica: Dict = field(default_factory=dict)

    def categorias(self) -> List[str]:
        return [s.categoria for s in self.sinais]

    def to_dict(self) -> Dict:
        return {
            "sinais": [
                {"categoria": s.categoria, "rotulo": s.rotulo,
                 "termos": s.termos, "intensidade": round(s.intensidade, 2)}
                for s in self.sinais
            ],
            "nivel_alerta": self.nivel_alerta,
            "resumo_llm": self.resumo_llm,
            "backend_llm": self.backend_llm,
            "sentimento": self.sentimento,
            "termos_chave": self.termos_chave,
            "backend_texto": self.backend_texto,
            "acustica": self.acustica,
        }


def _normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def _analisar_lexico(transcript: str) -> List[SinalDetectado]:
    norm = _normalizar(transcript)
    sinais: List[SinalDetectado] = []
    for categoria, termos in _LEXICO.items():
        achados = [t for t in termos if re.search(r"\b" + re.escape(t) + r"\b", norm)]
        if achados:
            intensidade = min(len(achados) / 3.0, 1.0)  # 3+ termos = intensidade máxima
            sinais.append(SinalDetectado(
                categoria=categoria, rotulo=_ROTULOS[categoria],
                termos=achados, intensidade=intensidade,
            ))
    return sinais


def _nivel(sinais: List[SinalDetectado]) -> str:
    # Violência doméstica eleva direto para vermelho.
    if any(s.categoria == "violencia_domestica" for s in sinais):
        return "vermelho"
    if any(s.intensidade >= 0.66 for s in sinais):
        return "vermelho"
    if sinais:
        return "amarelo"
    return "verde"


def _resumo_llm(transcript: str, sinais: List[SinalDetectado]) -> (str, str):
    """Enriquecimento opcional com a LLM do projeto; falha em silêncio p/ mock."""
    try:
        from src.llm.client import LLMClient
        from src.assistant.safety import guardrails
    except Exception:
        return "", ""
    cats = ", ".join(s.rotulo for s in sinais) or "nenhum sinal relevante"
    system = (
        "Você é um assistente de triagem em saúde da mulher. Resuma, em 2 frases, "
        "os sinais psicológicos observados no relato, de forma acolhedora e sem "
        "diagnosticar. Não prescreva."
    )
    prompt = (
        f"Relato da paciente: \"{transcript}\"\n"
        f"Sinais de triagem detectados: {cats}.\n"
        "Escreva um resumo curto para a equipe."
    )
    try:
        cli = LLMClient(backend="auto")
        # O mock do projeto é específico de predição SRAG; só enriquecemos com a
        # LLM real (Ollama). No modo offline fica a análise por léxico.
        if cli.active_backend == "mock":
            return "", "mock"
        txt = guardrails.sanitizar_saida(cli.generate(prompt, system=system))
        return txt, cli.active_backend
    except Exception:
        return "", ""


def _azure_text_analytics(transcript: str):
    """
    Azure AI Language (Text Analytics): sentimento + termos-chave.
    Segundo serviço de nuvem, além do Speech. Requer AZURE_LANGUAGE_KEY/ENDPOINT.
    Retorna (sentimento, termos, "azure") ou None se indisponível.
    """
    from .. import config
    if not (config.AZURE_LANGUAGE_KEY and config.AZURE_LANGUAGE_ENDPOINT):
        return None
    try:
        from azure.ai.textanalytics import TextAnalyticsClient
        from azure.core.credentials import AzureKeyCredential
        cli = TextAnalyticsClient(
            endpoint=config.AZURE_LANGUAGE_ENDPOINT,
            credential=AzureKeyCredential(config.AZURE_LANGUAGE_KEY),
        )
        sent = cli.analyze_sentiment([transcript], language="pt")[0]
        kp = cli.extract_key_phrases([transcript], language="pt")[0]
        sentimento = "" if sent.is_error else sent.sentiment
        termos = [] if kp.is_error else list(kp.key_phrases)[:10]
        return sentimento, termos, "azure"
    except Exception:
        return None


def _sentimento_lexico(sinais: List[SinalDetectado]) -> str:
    """Fallback offline: deriva um sentimento grosseiro dos sinais detectados."""
    if any(s.categoria == "violencia_domestica" for s in sinais):
        return "negative"
    if sinais:
        return "negative"
    return "neutral"


def analisar(transcript: str, usar_llm: bool = True,
             caminho_audio: Optional[str] = None) -> AnaliseAudio:
    sinais = _analisar_lexico(transcript)
    nivel = _nivel(sinais)

    resumo, backend = ("", "")
    if usar_llm:
        resumo, backend = _resumo_llm(transcript, sinais)

    # Sentimento + termos-chave: Azure Text Analytics, com fallback léxico.
    az = _azure_text_analytics(transcript)
    if az is not None:
        sentimento, termos, backend_texto = az
    else:
        sentimento = _sentimento_lexico(sinais)
        termos = sorted({t for s in sinais for t in s.termos})
        backend_texto = "lexico"

    # Características acústicas (quando o áudio é informado).
    acustica: Dict = {}
    if caminho_audio:
        try:
            from .acoustics import extrair
            feats = extrair(caminho_audio)
            acustica = feats.to_dict()
            # sinais acústicos elevam um quadro verde para amarelo.
            if feats.sinais and nivel == "verde":
                nivel = "amarelo"
        except Exception:
            acustica = {}

    return AnaliseAudio(
        transcript=transcript, sinais=sinais, nivel_alerta=nivel,
        resumo_llm=resumo, backend_llm=backend,
        sentimento=sentimento, termos_chave=termos, backend_texto=backend_texto,
        acustica=acustica,
    )
