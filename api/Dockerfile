FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV RAG_API_PORT=8088
EXPOSE 8088

# Default: run the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8088"]
