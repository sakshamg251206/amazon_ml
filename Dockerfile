# Self-contained demo image: generates the synthetic data, trains the model and builds the
# index at build time, so the container starts serving in seconds.
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8000
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY resolver ./resolver
RUN python -m resolver demo && rm -rf /root/.cache

RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/api/health')"
CMD ["sh", "-c", "uvicorn resolver.api.app:app --host 0.0.0.0 --port ${PORT} --proxy-headers"]
