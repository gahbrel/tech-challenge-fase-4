# Plano — Tech Challenge Fase 4

**FIAP Pós-Tech — Machine Learning Engineering**
**Tema:** Monitoramento multimodal em saúde da mulher (vídeo + áudio + texto)

---

## 1. Objetivo

Ampliar o assistente clínico das fases anteriores para **monitoramento
multimodal** voltado à **saúde da mulher**, fundindo vídeo, áudio e texto para
detectar precocemente riscos e acionar a equipe especializada. Reaproveitamos o
"motor" da Fase 3 (fluxo de decisão, guardrails e auditoria) como **camada de
fusão** das novas modalidades.

## 2. Escolhas do enunciado

- **Funcionalidades (≥2):** análise de **vídeo** clínico + processamento de
  **áudio/voz** de consultas.
- **Objetivos (≥3):** (1) detecção precoce de risco materno/ginecológico;
  (2) monitoramento do bem-estar psicológico feminino; (3) uso de **serviços em
  nuvem** (Azure) para ampliar o processamento.
- **YOLOv8:** detecção de **instrumentos cirúrgicos ginecológicos**.

## 3. Decisão de domínio

O enunciado assume um assistente já especializado em saúde da mulher; as fases
1–3 eram de SRAG (respiratório). **Pivotamos o tema para saúde da mulher** nesta
fase, mas **mantendo a arquitetura** da Fase 3 (LangGraph, guardrails,
auditoria) — que passa a coordenar as modalidades. O motor é o mesmo; o conteúdo
é de saúde da mulher.

## 4. Como atendemos os requisitos

| Requisito | Implementação | Onde |
|-----------|---------------|------|
| Análise de vídeo (YOLOv8) | detecção de instrumentos + relatório de desvios | `src/multimodal/video/` |
| Modelo YOLOv8 customizado | treino real (Colab/GPU) + demo sintético offline | `video/train_yolo.py` |
| Relatórios automáticos | achados + anomalias + nível de alerta | `video/report.py`, `audio/report.py` |
| Análise de áudio | transcrição + triagem psicológica | `src/multimodal/audio/` |
| Serviços em nuvem (Azure) | Azure Speech F0 (STT) + fallback Whisper/mock | `audio/transcribe.py` |
| Fusão multimodal | LangGraph que funde vídeo+áudio e alerta | `fusion/multimodal_graph.py` |
| Alerta em tempo real à equipe | nó de alerta + auditoria | `fusion/` + auditoria da Fase 3 |
| Privacidade/segurança | dados sintéticos/anonimizados; guardrails; validação humana | transversal |

## 5. Estratégia "roda em qualquer máquina"

Mesma filosofia das fases anteriores (real + demo/fallback):
- **Vídeo:** treino real no Colab (GPU) com dataset público de instrumentos; o
  `best.pt` é versionado e a **inferência roda no CPU** de todos. Modo demo
  treina offline com dataset sintético.
- **Áudio:** Azure F0 (nuvem) → Whisper (offline) → mock (transcript ao lado do
  áudio). A demo roda sem conta Azure.
- **Fusão:** LangGraph com fallback para runner Python puro.

## 6. Dados

- **Vídeo:** dataset público de instrumentos cirúrgicos em formato YOLO (ex.:
  Roboflow Universe / m2cai16-tool) para o treino real; dataset sintético para a
  demo. Nenhum dado real de paciente.
- **Áudio:** clipes públicos/atuados de consulta; a validade é do *pipeline*, não
  clínica. Transcripts de demonstração incluídos.

## 7. Entregáveis

- Código em `src/multimodal/` + `run_fase4.py`
- Relatório técnico `docs/relatorio_fase4.md`
- Notebook `notebooks/09_multimodal_fase4.ipynb`
- Testes `tests/test_multimodal.py`
- Roteiro do vídeo `docs/roteiro_video_fase4.md`
