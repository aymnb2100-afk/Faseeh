# Faseeh analysis server on this computer, on its NVIDIA graphics card (Docker Desktop + WSL 2).
# The image holds only the tools. The server code, settings and calibration are taken from GitHub when the
# container starts and kept up to date by supervisor.py, so nothing here needs changing when the code changes.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# PyTorch built for NVIDIA cards (CUDA 12.6 supports the RTX 20xx series and newer)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir "quran-muaalem[engine]" "transformers==4.55.2" levenshtein soundfile fastapi uvicorn

RUN useradd -m -u 1000 faseeh && mkdir -p /src /cache && chown faseeh:faseeh /src /cache
USER faseeh
ENV HF_HOME=/cache/huggingface PYTHONUNBUFFERED=1 PIP_USER=1 PATH=/home/faseeh/.local/bin:$PATH \
    FASEEH_MAX_SECONDS=330 \
    FASEEH_REPO=https://github.com/aymnb2100-afk/Faseeh.git FASEEH_BRANCH=main
COPY --chown=faseeh boot.sh /home/faseeh/boot.sh
RUN sed -i 's/\r$//' /home/faseeh/boot.sh
CMD ["sh", "/home/faseeh/boot.sh"]
