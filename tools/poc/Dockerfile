# Sandbox d'exécution des notebooks de POC : Linux, CPU seul, utilisateur sans privilège, aucun secret.
# Bibliothèques courantes préinstallées (moins de téléchargements et d'erreurs par notebook) ; les versions
# réellement utilisées sont relevées par `pip freeze` après chaque exécution (requirements.txt livré avec le POC).
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 git build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir \
      nbclient nbformat ipykernel \
      numpy pandas pyarrow scipy scikit-learn matplotlib tqdm \
      datasets huggingface_hub transformers sentence-transformers accelerate torchcodec \
      soundfile openai httpx \
      xgboost lightgbm networkx

RUN useradd --create-home --uid 1000 poc
USER poc
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 MPLBACKEND=Agg
WORKDIR /travail
