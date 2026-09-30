# API image: FastAPI + the screening pipeline.
#   docker build -t due-diligence-agent .
#   docker run -p 8000:8000 --env-file .env -v dd-data:/app/data due-diligence-agent
# The volume keeps the SQLite store (audit trail, labels, jobs) and the sanctions export.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first (pinned lock file), so code changes don't reinstall them.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Run as an unprivileged user; data/ and reports/ are the only writable paths.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p data reports \
    && chown -R app:app data reports
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]
