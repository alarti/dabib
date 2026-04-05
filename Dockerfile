# Dockerfile
FROM python:3.11-slim

WORKDIR /app

ENV LLM_SERVER_URL=http://llm:8000/v1/chat/completions
ENV SEARXNG_URL=http://searxng:8081/search
ENV MEMORY_DB_PATH=/data/biological_memory
ENV USER_PROFILE_FILE=/data/user_profile.txt
ENV EPISODE_LOG_FILE=/data/episodes_log.jsonl

# Copy dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY app/ app/
COPY .env.example .env

# Copy data mount (for ChromaDB, profile, logs)
VOLUME ["/data"]

# Expose DABIB port
EXPOSE 8000

# Entrypoint
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
