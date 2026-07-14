FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=3000

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY app ./app
COPY public ./public
COPY knowledge_base ./knowledge_base
COPY storage ./storage
COPY alembic.ini .
COPY alembic ./alembic

EXPOSE 3000

CMD ["sh", "-c", "alembic upgrade head && exec python -m uvicorn app.fastapi_main:app --host 0.0.0.0 --port 3000"]
