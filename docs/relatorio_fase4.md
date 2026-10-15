# Relatório Técnico — Tech Challenge Fase 4
## Monitoramento Multimodal em Saúde da Mulher (vídeo + áudio + texto)
### FIAP Pós-Tech — Machine Learning Engineering

---

## 1. Introdução

As fases anteriores construíram, para uma rede hospitalar, modelos de
classificação (Fase 1), otimização por algoritmo genético e LLM explicativa
(Fase 2) e um assistente clínico com fine-tuning, RAG e fluxo de decisão
(Fase 3). A Fase 4 dá o próximo passo: **monitoramento multimodal** voltado à
**saúde da mulher**, fundindo **vídeo, áudio e texto** para identificar
precocemente sinais de risco e acionar a equipe especializada.

O tema é pivotado para saúde da mulher (as fases 1–3 eram de SRAG), mas a
**arquitetura de decisão da Fase 3 é reaproveitada** como camada de fusão —
mantendo guardrails, auditoria e o princípio de que o sistema é **apoio à
decisão com validação humana**, nunca um substituto do profissional.

Escolhas do enunciado: funcionalidades de **vídeo** e **áudio**; objetivos de
**risco materno/ginecológico**, **bem-estar psicológico** e **uso de nuvem
(Azure)**; YOLOv8 para **instrumentos cirúrgicos ginecológicos**.

---

## 2. Fluxo multimodal

```
 VÍDEO (cirurgia)            ÁUDIO (consulta)             TEXTO/contexto (Fase 3)
      │                           │                             │
  YOLOv8 detect              Azure STT (F0)                 assistente/RAG
  (instrumentos)             → transcrição                  (reaproveitado)
      │                       → triagem psicológica              │
  relatório vídeo            relatório áudio                     │
      └───────────────┬───────────┴───────────────────────────── ┘
                      ▼
        FUSÃO (LangGraph): receber_video → receber_audio → fundir
                      → (risco?) → emitir_alerta | registrar_ok → consolidar
                      → auditoria (Fase 3)
```

A fusão combina os níveis de alerta das modalidades (o maior prevalece), agrega
os achados e, em risco amarelo/vermelho, emite o alerta para a equipe. Todo o
percurso é auditável (trilha de nós + log estruturado).

---

## 3. Modelos e técnicas por tipo de dado

### 3.1 Vídeo — YOLOv8 (`src/multimodal/video/`)

- **Modelo:** YOLOv8 (Ultralytics) para detecção de instrumentos cirúrgicos
  ginecológicos (bisturi, pinça, tesoura, afastador).
- **Treino real** (`train_yolo.py --mode real`): sobre um dataset público em
  formato YOLO (Roboflow Universe / m2cai16-tool). Recomendado no **Google Colab**
  (GPU grátis); o `best.pt` (~6 MB) é versionado e a inferência roda em CPU.
- **Treino demo** (`--mode demo`): gera um dataset sintético de instrumentos e
  treina o YOLOv8n offline, produzindo um `best.pt` real. Na demonstração deste
  relatório: **mAP50 = 0,96** (12 épocas, 320px, dataset sintético).
- **Inferência** (`detect.py`): roda quadro a quadro, devolve detecções
  estruturadas (classe, confiança, bbox, tempo) e salva o **vídeo anotado**.
- **Relatório** (`report.py`): aplica regras de anomalia/desvio — instrumento
  fora do protocolo, baixa cobertura temporal (possível obstrução do campo),
  ausência de instrumentos — e define o nível de alerta (verde/amarelo/vermelho).

### 3.2 Áudio — transcrição + triagem (`src/multimodal/audio/`)

- **Transcrição** (`transcribe.py`): backends intercambiáveis — **Azure AI
  Speech (F0)** para o caminho de nuvem, **Whisper** local como alternativa
  offline, e **mock** (transcript ao lado do áudio) para a demo. Resolução
  automática: Azure se houver chave → Whisper se instalado → mock.
- **Triagem** (`analyze.py`): análise por **léxico ponderado** de quatro
  categorias — ansiedade gestacional, depressão pós-parto, fadiga hormonal e
  possível violência doméstica. É **determinística, offline e explicável**: cada
  sinal registra os termos que o dispararam. Opcionalmente enriquece com a LLM do
  projeto (só com Ollama real). Violência ou intensidade alta elevam o alerta.
- **Relatório** (`report.py`): consolida transcrição + sinais + nível + aviso de
  triagem (não é diagnóstico; exige validação profissional).

### 3.3 Texto/contexto — Fase 3 reaproveitada

O assistente, o RAG e a base de pacientes da Fase 3 permanecem disponíveis como
camada de contexto e são o alicerce do fluxo de decisão reusado na fusão.

---

## 4. Fusão multimodal (`src/multimodal/fusion/`)

Implementada com **LangGraph** (`multimodal_graph.py`), no mesmo padrão da Fase 3
(nós de estado + roteamento condicional), com **fallback para runner Python puro**
quando o LangGraph não está instalado — garantindo portabilidade.

Nós: `receber_video` → `receber_audio` → `fundir` → (roteamento por risco) →
`emitir_alerta` **ou** `registrar_ok` → `consolidar`. Cada emissão de alerta ou
registro gera um `AuditEvent` (auditoria da Fase 3), tornando a fusão rastreável.

---

## 5. Segurança, privacidade e explainability

- **Dados:** sintéticos/anonimizados; nenhum dado real de paciente. O vídeo usa
  dataset sintético (demo) ou público (treino); o áudio usa clipes/transcripts de
  demonstração.
- **Validação humana:** todo relatório e alerta traz o aviso de que é apoio à
  decisão e exige validação da equipe. Reaproveita os guardrails da Fase 3.
- **Explainability:** o vídeo cita os instrumentos/anomalias detectados; o áudio
  cita os **termos** que dispararam cada sinal; a fusão lista os achados por
  modalidade. Nada é "caixa-preta".
- **Auditoria:** log estruturado de cada decisão de fusão.
- **Tema sensível:** violência doméstica e saúde mental são tratados como
  **triagem de encaminhamento**, com orientação de acionar a rede de proteção —
  nunca como diagnóstico automático.

---

## 6. Resultados e exemplos de anomalias

Execução de demonstração (offline, `python run_fase4.py`):

- **Vídeo:** detector demo com **mAP50 = 0,96**; no clipe de cirurgia, os quatro
  instrumentos foram detectados e contabilizados; sem anomalias → alerta VERDE.
- **Áudio (cenário de triagem de violência):** transcrição via mock; a triagem
  detectou ansiedade, depressão pós-parto e **possível violência doméstica**
  (termos: "me empurrou", "tenho medo dele", "não posso sair", "ele controla",
  "grita comigo") → alerta VERMELHO.
- **Fusão:** risco combinado = **VERMELHO** (o maior nível prevalece) → **alerta
  multimodal emitido** para a equipe, com achados agregados e trilha auditável
  `receber_video → receber_audio → fundir → emitir_alerta → consolidar`.

**Métricas de avaliação (vídeo):** mAP50 e mAP50-95 do YOLOv8 sobre o conjunto de
validação. **Áudio:** a triagem é avaliada qualitativamente por cenário (os testes
verificam que violência → vermelho, ausência de sinais → verde, e que os termos
são registrados). 7 testes automatizados cobrem áudio e fusão.

**Limitações.** (i) O detector de demonstração usa dados sintéticos — o treino
real com o dataset público de instrumentos (Colab) é o caminho de produção;
(ii) a triagem por léxico é um *screening* explicável, não um classificador
clínico validado; (iii) a validade clínica dos áudios/vídeos de demo não é o foco
— o foco é o *pipeline* multimodal.

---

## 7. Reprodutibilidade

```bash
pip install -r requirements.txt -r requirements-fase4.txt

# fluxo multimodal completo (offline, dados de demonstração)
python run_fase4.py

# treinar o detector demo antes
python run_fase4.py --treinar

# testes
pytest tests/test_multimodal.py -v
```

Caminhos reais: treino do YOLOv8 no Colab (`train_yolo.py --mode real --data data.yaml`);
áudio via Azure (`AUDIO_BACKEND=azure` + chave/região F0) ou Whisper
(`AUDIO_BACKEND=whisper`); LLM via Ollama (`LLM_BACKEND=ollama`).

---

## 8. Conclusão

A Fase 4 entrega um pipeline **multimodal** coeso: detecção de instrumentos em
vídeo (YOLOv8), transcrição e triagem psicológica de áudio (Azure/Whisper) e uma
camada de **fusão em LangGraph** que combina as modalidades, decide o risco e
aciona a equipe — tudo com segurança, explainability e auditoria herdadas da
Fase 3. A arquitetura é plugável: trocando os backends, a mesma solução opera com
o detector treinado em dados reais, transcrição na nuvem (Azure F0) e a LLM
fine-tuned, sem alterar o fluxo de fusão.
