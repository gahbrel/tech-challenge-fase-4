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
 VÍDEO (cirurgia/consulta)   ÁUDIO (consulta)        SINAIS VITAIS (gestante)
  YOLOv8: instrumentos       Azure Speech → texto    série temporal
  YOLOv8-pose: postura       Text Analytics: sent.   Isolation Forest +
      │                      librosa: acústica       regras clínicas
  relatório vídeo+postura    relatório áudio         relatório vitais
      └──────────────┬───────────────┴───────────────────────┘
                     ▼
   FUSÃO (LangGraph): receber_video → receber_audio → receber_vitais
          → receber_postura → fundir → (risco?)
          → emitir_alerta | registrar_ok → consolidar → auditoria (Fase 3)
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
- **Análise postural** (`pose.py`): usa **YOLOv8-pose** (keypoints COCO) como
  alternativa prática e instalável ao **OpenPose** citado no enunciado — ambos
  fazem estimativa de pose 2D. Calcula marcadores explicáveis (movimento médio
  entre frames, assimetria de ombros/quadris, postura defensiva) para sinalizar
  rigidez/desconforto em fisioterapia pós-parto e sinais não-verbais em consultas.
  Caminho real em vídeo com pessoa; demo offline com keypoints sintéticos.

### 3.2 Áudio — transcrição + triagem + acústica (`src/multimodal/audio/`)

- **Transcrição** (`transcribe.py`): **Azure AI Speech (F0)** (nuvem), **Whisper**
  (offline) e **mock** (transcript ao lado do áudio). Resolução automática.
- **Triagem por léxico** (`analyze.py`): análise ponderada de quatro categorias —
  ansiedade gestacional, depressão pós-parto, fadiga hormonal e possível violência
  doméstica. Determinística, offline e **explicável** (registra os termos).
- **Azure Text Analytics** (`analyze.py`): **segundo serviço de nuvem**, extrai
  **sentimento** e **termos-chave** da transcrição. Fallback = sentimento/termos
  derivados do léxico (offline). Marca a integração de nuvem além do Speech.
- **Características acústicas** (`acoustics.py`): com **librosa**, extrai energia,
  tom (F0 e variação), ritmo de fala e proporção de pausas — analisa *como* a
  paciente fala, não só o que fala. Sinaliza fadiga/baixo ânimo e tensão/estresse;
  um sinal acústico eleva um quadro verde para amarelo.
- **Relatório** (`report.py`): consolida transcrição + sinais (léxico + acústicos)
  + sentimento + nível + aviso de triagem (não é diagnóstico).

### 3.3 Sinais vitais — detecção de anomalias (`src/multimodal/vitals/`)

Monitora séries temporais de sinais vitais maternos (pressão arterial, batimentos
fetais, FC materna, saturação) combinando **duas abordagens**:
- **Regras clínicas** (faixas de referência obstétricas): PA ≥ 140/90 →
  hipertensão gestacional; ≥ 160/110 → possível pré-eclâmpsia; BCF < 110 ou > 160
  → fora da faixa; < 100 → bradicardia fetal. Explicáveis e determinísticas.
- **Modelo de detecção de anomalias** (`Isolation Forest`, scikit-learn; z-score
  como alternativa) — captura desvios estatísticos para monitoramento preventivo.
Séries de demonstração são sintéticas, com anomalias injetadas (pico hipertensivo,
bradicardia fetal). Define o nível de alerta e entra na fusão.

### 3.4 Texto/contexto — Fase 3 reaproveitada

O assistente, o RAG e a base de pacientes da Fase 3 permanecem disponíveis como
camada de contexto e alicerce do fluxo de decisão reusado na fusão.

---

## 4. Fusão multimodal (`src/multimodal/fusion/`)

Implementada com **LangGraph** (`multimodal_graph.py`), no mesmo padrão da Fase 3
(nós de estado + roteamento condicional), com **fallback para runner Python puro**
quando o LangGraph não está instalado — garantindo portabilidade.

A fusão coordena **quatro modalidades**: vídeo (instrumentos + postura), áudio
(triagem + acústica), sinais vitais e — como contexto — a camada textual da
Fase 3. Nós: `receber_video` → `receber_audio` → `receber_vitais` →
`receber_postura` → `fundir` → (roteamento por risco) → `emitir_alerta` **ou**
`registrar_ok` → `consolidar`. A regra de fusão é o **maior nível entre as
modalidades**, com os achados agregados por modalidade. Cada emissão de alerta ou
registro gera um `AuditEvent` (auditoria da Fase 3), tornando a fusão rastreável.
A assinatura é retrocompatível: chamadas antigas (só vídeo + áudio) seguem válidas.

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
- **Áudio (cenário de fadiga pós-parto):** a acústica (librosa) detectou voz de
  baixa energia e muitas pausas → **fadiga/baixo ânimo**; o sentimento resultou
  **negative** (Azure Text Analytics ou fallback léxico).
- **Sinais vitais:** a série demo acionou **pré-eclâmpsia** (PA 169/116) por regra
  clínica e **bradicardia fetal** (BCF 98), com o Isolation Forest confirmando os
  desvios estatísticos → alerta VERMELHO.
- **Postura (YOLOv8-pose):** no cenário de desconforto, movimento reduzido +
  assimetria de ombros + postura defensiva → alerta AMARELO.
- **Fusão:** risco combinado = **VERMELHO** (o maior nível entre as modalidades
  prevalece) → **alerta multimodal** para a equipe, com achados agregados das
  quatro modalidades e trilha auditável `receber_video → receber_audio →
  receber_vitais → receber_postura → fundir → emitir_alerta → consolidar`.

**Métricas de avaliação (vídeo):** mAP50 e mAP50-95 do YOLOv8 sobre validação
(demo: mAP50 = 0,96). **Áudio/vitais/postura:** avaliação qualitativa por cenário,
coberta por testes automatizados (violência → vermelho, pré-eclâmpsia/bradicardia
→ vermelho, desconforto postural → amarelo, ausência de sinais → verde). Ao todo,
**17 testes** cobrem áudio, vitais, postura, Text Analytics e a fusão de 4 modalidades.

**Limitações.** (i) Detector e séries de demonstração usam dados sintéticos — os
caminhos reais (dataset público de instrumentos no Colab; vídeo com pessoa para a
pose; Azure com chave; monitores reais para os sinais vitais) são plugáveis;
(ii) as triagens por léxico/acústica/pose são *screening* explicável, não
classificadores clínicos validados; (iii) o foco é o *pipeline* multimodal, não a
validade clínica dos exemplos de demo.

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
