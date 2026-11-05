# Faseeh analysis server on a Hugging Face Space (Docker, free CPU hardware)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface \
    FASEEH_MAX_SECONDS=90 OMP_NUM_THREADS=2
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --user -r requirements.txt

# downloads the model into the image, so the Space starts faster
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('obadx/muaalem-model-v3_2')"

COPY --chown=user . .
EXPOSE 7860
CMD ["uvicorn", "faseeh_server:app", "--host", "0.0.0.0", "--port", "7860"]
