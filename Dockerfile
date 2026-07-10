FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=3000

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY public ./public
COPY knowledge_base ./knowledge_base
COPY storage ./storage

EXPOSE 3000

CMD ["python", "-m", "uvicorn", "app.fastapi_main:app", "--host", "0.0.0.0", "--port", "3000"]
