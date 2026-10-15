"""Gera notebooks/colab_treino_yolov8_fase4.ipynb (treino real no Google Colab)."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md("""# Fase 4 — Treino real do YOLOv8 (Google Colab)

Treina o detector de **instrumentos cirúrgicos ginecológicos** num dataset
público (Roboflow), usando a **GPU grátis do Colab**, e gera o `best.pt` para
usar no projeto (a inferência depois roda em CPU na máquina de qualquer membro).

### Antes de começar
1. Menu **Ambiente de execução → Alterar tipo de ambiente → GPU (T4)**.
2. Crie uma conta grátis no **Roboflow** (https://roboflow.com) para pegar sua
   `API_KEY` (em *Settings → API*).
3. Escolha um dataset de instrumentos cirúrgicos no **Roboflow Universe**
   (https://universe.roboflow.com — busque por *surgical instruments*). Na página
   do dataset, clique em **Download this Dataset → YOLOv8 → show download code**;
   ele te dá o `workspace`, o `project` e a `version` para colar abaixo.

> Alternativas de dataset conhecidas: *m2cai16-tool* (7 instrumentos) e vários
> *surgical-tools* no Universe. Qualquer um em formato YOLOv8 serve.""")

md("## 1. Checar a GPU")
code("!nvidia-smi")

md("## 2. Instalar dependências")
code("!pip -q install ultralytics roboflow")

md("""## 3. Baixar o dataset (Roboflow)

Cole os valores do *show download code* do dataset escolhido. O `download("yolov8")`
já baixa no formato certo e cria o `data.yaml`.""")
code('''from roboflow import Roboflow

API_KEY   = "SUA_API_KEY"        # Roboflow → Settings → API
WORKSPACE = "SEU_WORKSPACE"      # do "show download code"
PROJECT   = "SEU_PROJECT"        # do "show download code"
VERSION   = 1                     # do "show download code"

rf = Roboflow(api_key=API_KEY)
dataset = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download("yolov8")
DATA_YAML = f"{dataset.location}/data.yaml"
print("data.yaml em:", DATA_YAML)

# conferir as classes do dataset (importante para o passo 7)
import yaml
print("classes:", yaml.safe_load(open(DATA_YAML)).get("names"))''')

md("""## 4. Treinar o YOLOv8

`yolov8n` é rápido e leve (bom para CPU depois). Para mais precisão, troque por
`yolov8s`. Ajuste `epochs` conforme o tempo disponível.""")
code('''from ultralytics import YOLO

modelo = YOLO("yolov8n.pt")          # ou "yolov8s.pt"
resultados = modelo.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=16,
    project="runs_fase4",
    name="instrumentos",
    plots=True,
)''')

md("## 5. Métricas e gráficos do treino")
code('''import glob
from IPython.display import Image, display

save_dir = "runs_fase4/instrumentos"
for img in ["results.png", "confusion_matrix.png", "val_batch0_pred.jpg"]:
    caminhos = glob.glob(f"{save_dir}/{img}")
    if caminhos:
        display(Image(filename=caminhos[0]))

# métricas finais
metricas = modelo.val(data=DATA_YAML)
print("mAP50   :", round(float(metricas.box.map50), 4))
print("mAP50-95:", round(float(metricas.box.map), 4))''')

md("## 6. Teste rápido de inferência")
code('''import glob
amostras = glob.glob(f"{dataset.location}/valid/images/*.jpg")[:2]
if amostras:
    pred = YOLO(f"{save_dir}/weights/best.pt").predict(amostras, save=True, conf=0.25)
    for p in pred:
        display(Image(filename=p.path.replace("valid/images", "").rsplit("/",1)[0] + "/" ))  # opcional
    print("Predições salvas em runs de predict/ do Ultralytics.")''')

md("""## 7. Baixar o `best.pt` e usar no projeto

Baixe o arquivo e coloque em **`results/multimodal/video/best.pt`** no repositório.
A partir daí, `src/multimodal/video/detect.py` usa automaticamente esse peso
treinado (em vez do modelo demo), rodando em CPU.""")
code('''from google.colab import files
files.download(f"{save_dir}/weights/best.pt")''')

md("""### ⚠️ Ajuste importante após treinar com dados reais

O dataset real tem **nomes de classe próprios** (ex.: `scalpel`, `forceps`,
`scissors`, `needle_driver`…), diferentes dos nomes do demo. Depois de colocar o
`best.pt` no projeto, atualize a lista de instrumentos esperados em
`src/multimodal/video/report.py`:

```python
INSTRUMENTOS_ESPERADOS = {"scalpel", "forceps", "scissors", "needle_driver"}  # use as classes do SEU dataset
```

Assim a regra de "instrumento não previsto no protocolo" passa a fazer sentido
com as classes reais. O restante do pipeline (detecção, relatório, fusão) não
muda.""")

nb["cells"] = cells
out = ROOT / "notebooks" / "colab_treino_yolov8_fase4.ipynb"
nbf.write(nb, str(out))
print("[ok]", out)
