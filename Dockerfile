FROM python:3.13-slim
WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && pip install awscli \
    && pip install -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

CMD ["python3", "app.py"]
