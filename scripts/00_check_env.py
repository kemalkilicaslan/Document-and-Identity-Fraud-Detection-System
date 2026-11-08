"""Step 00: check packages and Tesseract, extract the SIDTD archives."""
import importlib
import shutil
import zipfile

from _bootstrap import ROOT, done, step

from docfraud.config import load_config, path

step("00 · Environment check")
missing = []
for mod in ["torch", "torchvision", "timm", "PIL", "numpy", "pandas", "pyarrow", "sklearn",
            "lightgbm", "duckdb", "faker", "yaml", "pytesseract", "networkx", "matplotlib",
            "markdown", "tabulate", "fastapi", "uvicorn", "streamlit", "onnx", "onnxruntime", "onnxscript"]:
    try:
        importlib.import_module(mod)
    except ImportError:
        missing.append(mod)
if missing:
    raise SystemExit(f"Missing packages: {missing}\nRun: pip install -r requirements.txt")

import torch  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"torch {torch.__version__}, device: {dev}")

if not shutil.which("tesseract"):
    raise SystemExit("Tesseract not found. macOS: brew install tesseract")
print("tesseract:", shutil.which("tesseract"))

cfg = load_config()
raw, interim = path("raw"), path("interim")
for z in cfg["sidtd"]["zips"]:
    name = z.replace(".zip", "")
    target = interim / name
    if target.exists() and any(target.iterdir()):
        print(f"{name}: already extracted")
        continue
    src = raw / z
    if not src.exists():
        raise SystemExit(f"{src} not found. Download it from http://datasets.cvc.uab.es/SIDTD/{z}")
    print(f"extracting {z} ... (this can take a few minutes)")
    with zipfile.ZipFile(src) as zf:
        zf.extractall(interim)

for sub in ["templates/Images/reals", "templates/Annotations/reals",
            "clips_cropped/Images/reals", "clips_cropped/Images/fakes",
            "clips_cropped/annotations/fakes"]:
    n = sum(1 for _ in (interim / sub).iterdir())
    print(f"{sub:40s} {n:>7} files")

free_gb = shutil.disk_usage(ROOT).free / 1e9
print(f"free disk: {free_gb:.0f} GB (patches need ~3 GB)")
done()
