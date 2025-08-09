FROM python:3.11-slim

WORKDIR /work/api

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API source code
COPY api/ .

EXPOSE 8088

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8088", "--reload"]
