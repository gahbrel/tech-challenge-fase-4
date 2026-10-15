# Roteiro do Vídeo — Fase 4 (até 15 min)

**Objetivo:** demonstrar o processamento multimodal (vídeo + áudio), a detecção e
resposta a anomalias, a integração com o Azure e o fluxo final de alerta à equipe.

Dica: rode a demo offline com `LLM_BACKEND=mock`. Para mostrar o Azure de verdade,
configure `AZURE_SPEECH_KEY`/`AZURE_SPEECH_REGION` (tier F0) e `AUDIO_BACKEND=azure`.

---

## 0. Abertura (0:00–1:30)
- Grupo + recap: Fases 1–3 (classificação, AG+LLM, assistente com fine-tuning).
- Fase 4: monitoramento **multimodal** em **saúde da mulher** — vídeo + áudio +
  texto, fundidos para alertar a equipe. Mostrar o diagrama do fluxo (relatório).
- Reforçar: apoio à decisão com validação humana; dados sintéticos/anonimizados.

## 1. Análise de vídeo — YOLOv8 (1:30–5:00)
- Explicar o treino: `train_yolo.py` — real no Colab com dataset público de
  instrumentos; demo offline com dataset sintético. Mostrar a métrica (mAP50).
- Rodar `python -m src.multimodal.video.detect data/multimodal/video/demo_cirurgia.mp4`.
- Abrir o **vídeo anotado** (`results/multimodal/video/..._anotado.mp4`) — as
  caixas dos instrumentos.
- Mostrar o **relatório especializado** (instrumentos, contagem, anomalias, nível).

## 2. Análise de áudio + Azure (5:00–9:00)
- Explicar os backends: Azure F0 (nuvem) / Whisper (offline) / mock.
- **Integração Azure:** mostrar o recurso F0 no portal e rodar com
  `AUDIO_BACKEND=azure` numa consulta (transcrição real). Explicar o tier grátis.
- Rodar a triagem no cenário de **violência doméstica**: mostrar os **termos
  detectados** (explainability) e o alerta VERMELHO.
- Mostrar também o cenário de **pós-parto** (ansiedade/depressão).

## 3. Fusão multimodal + alerta (9:00–12:30)
- Rodar `python run_fase4.py` — mostrar o fluxo 1→4.
- Destacar a **fusão LangGraph**: como vídeo + áudio viram um **risco combinado**,
  a **trilha** dos nós e o **alerta** final à equipe.
- Abrir a **auditoria** (log estruturado) — rastreabilidade da decisão.

## 4. Segurança, privacidade e limitações (12:30–14:00)
- Validação humana obrigatória; tema sensível (violência/saúde mental) tratado
  como triagem/encaminhamento.
- Dados sintéticos/anonimizados; explainability em todas as modalidades.

## 5. Encerramento (14:00–15:00)
- Rodar `pytest tests/test_multimodal.py -v` (testes passando).
- Fechar: arquitetura plugável — troca de backends ativa detector real, Azure e
  LLM fine-tuned sem mudar o fluxo de fusão. Herda guardrails/auditoria da Fase 3.

---

### Checklist de gravação
- [ ] Terminal com fonte legível; `LLM_BACKEND=mock` exportado (ou Azure configurado).
- [ ] Vídeo anotado e relatórios abertos.
- [ ] Recurso Azure F0 pronto (se for mostrar a nuvem ao vivo).
- [ ] Upload final no YouTube/Vimeo (público ou não listado).
