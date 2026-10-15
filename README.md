# Fase 4 — Monitoramento Multimodal em Saúde da Mulher

Pipeline **multimodal** (vídeo + áudio + texto) que detecta sinais de risco em
saúde da mulher e aciona a equipe especializada. Construído sobre as fases
anteriores: reaproveita o fluxo de decisão, os guardrails e a auditoria da
Fase 3 como **camada de fusão** das novas modalidades.

> Ferramenta de **apoio à decisão**. Não prescreve, não diagnostica e não
> substitui a avaliação do profissional responsável.

---

## O que foi entregue

| Requisito do enunciado | Implementação |
|------------------------|---------------|
| Análise de vídeo (YOLOv8) | detecção de instrumentos cirúrgicos + relatório de desvios — `src/multimodal/video/` |
| YOLOv8 customizado | treino real (Colab/GPU) + demo sintético offline — `video/train_yolo.py` |
| Análise de áudio | transcrição + triagem psicológica — `src/multimodal/audio/` |
| Serviços em nuvem (Azure) | Azure Speech F0 + fallback Whisper/mock — `audio/transcribe.py` |
| Fusão multimodal (LangGraph) | vídeo + áudio → risco combinado → alerta — `fusion/multimodal_graph.py` |
| Relatórios/alertas + auditoria | níveis de alerta + trilha auditável (herda a Fase 3) |

**Funcionalidades escolhidas:** vídeo + áudio.
**Objetivos escolhidos:** risco materno/ginecológico · bem-estar psicológico · uso de nuvem (Azure).
**YOLOv8:** detecção de instrumentos cirúrgicos ginecológicos.

---

## Estrutura

```
src/multimodal/
├── video/
│   ├── train_yolo.py     # treino YOLOv8 (real no Colab + demo sintético offline)
│   ├── detect.py         # inferência: vídeo → detecções + vídeo anotado
│   └── report.py         # relatório especializado (desvios, nível de alerta)
├── audio/
│   ├── transcribe.py     # Azure Speech F0 + Whisper (offline) + mock
│   ├── analyze.py        # triagem psicológica por léxico (termos rastreados)
│   └── report.py         # relatório especializado de áudio
├── fusion/
│   └── multimodal_graph.py  # LangGraph que funde vídeo+áudio e emite o alerta
└── config.py
data/multimodal/{video,audio}/   # dados de demonstração
results/multimodal/              # vídeos anotados, relatórios
run_fase4.py                     # orquestrador (roda o fluxo completo)
tests/test_multimodal.py
notebooks/
├── 09_multimodal_fase4.ipynb          # demonstração do pipeline
└── colab_treino_yolov8_fase4.ipynb    # treino real do YOLOv8 (Google Colab)
```

---

## Requisitos

- **Python 3.11 ou 3.12** (o 3.13+ ainda não tem wheels de algumas libs; evitar).
- Dependências em `requirements.txt` + `requirements-fase4.txt`.
- Opcionais: conta Azure (tier F0) para a transcrição na nuvem; GPU/Colab para o
  treino real do YOLOv8.

---

## Como executar (offline, sem GPU nem conta Azure)

```bash
# ambiente virtual com Python 3.12
py -3.12 -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-fase4.txt

# fluxo multimodal completo, com os dados de demonstração
python run_fase4.py

# treina o detector demo antes (gera results/multimodal/video/best.pt)
python run_fase4.py --treinar

# testes
pytest tests/test_multimodal.py -v
```

A demo roda **offline**: o áudio cai no backend `mock` (transcript ao lado do
arquivo) e o vídeo usa o detector demo. Os caminhos reais são plugáveis.

---

## Backends reais (plugáveis)

### Áudio com Azure Speech (tier gratuito F0)

1. Crie uma conta gratuita no Azure e um recurso **Speech** no tier **F0**
   (5 h/mês de transcrição, sem cobrança por excedente).
2. Pegue a **chave** e a **região** e exporte:

```bash
export AZURE_SPEECH_KEY="sua_chave"
export AZURE_SPEECH_REGION="brazilsouth"
export AUDIO_BACKEND=azure
python run_fase4.py
```

Sem Azure: `AUDIO_BACKEND=whisper` (offline, requer `openai-whisper`) ou `mock` (padrão).

### Treino real do YOLOv8 (Google Colab)

Abra `notebooks/colab_treino_yolov8_fase4.ipynb` no Colab (runtime **GPU**),
baixe um dataset público de instrumentos do Roboflow Universe e treine. Coloque o
`best.pt` gerado em `results/multimodal/video/best.pt` — o `detect.py` passa a usar
o modelo treinado automaticamente (inferência em CPU).

> Após treinar com dados reais, atualize `INSTRUMENTOS_ESPERADOS` em
> `video/report.py` com as classes do dataset escolhido.

---

## Fluxo multimodal

```
 VÍDEO (cirurgia)        ÁUDIO (consulta)           TEXTO/contexto (Fase 3)
   YOLOv8 detect         Azure STT → triagem        assistente / RAG
      │                       │                           │
 relatório vídeo        relatório áudio                   │
      └──────────┬────────────┴───────────────────────────┘
                 ▼
   FUSÃO (LangGraph): receber_video → receber_audio → fundir
         → (risco?) → emitir_alerta | registrar_ok → consolidar → auditoria
```

A fusão combina os níveis de alerta (o maior prevalece), agrega os achados e, em
risco amarelo/vermelho, emite o alerta à equipe — tudo rastreável.

---

## Segurança, privacidade e explainability

- **Dados** sintéticos/anonimizados; nenhum dado real de paciente.
- **Validação humana** obrigatória; todo relatório/alerta traz o aviso.
- **Explainability:** vídeo cita instrumentos/anomalias; áudio cita os **termos**
  que dispararam cada sinal; a fusão lista os achados por modalidade.
- **Tema sensível** (violência/saúde mental) tratado como triagem/encaminhamento,
  nunca diagnóstico automático.

---

## Entregáveis

- Código em `src/multimodal/` + `run_fase4.py`
- Relatório técnico: `docs/relatorio_fase4.md`
- Plano: `docs/plano_fase4.md`
- Roteiro do vídeo: `docs/roteiro_video_fase4.md`
- Notebooks: demonstração (`09_...`) e treino no Colab (`colab_...`)
- Testes: `tests/test_multimodal.py`