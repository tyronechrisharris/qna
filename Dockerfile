FROM python:3.9-slim-buster

# Set environment variables
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    postgresql-client \
    redis-tools \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download and install spaCy language model
RUN python -m spacy download en_core_web_sm
RUN python -m spacy_experimental.coref.download en

# Copy the application code
COPY . .

# Expose ports for Redis and PostgreSQL (optional, if you want to access them externally)
# EXPOSE 6379  # Redis default port
# EXPOSE 5432  # PostgreSQL default port

# Define commands to start Redis and PostgreSQL servers
CMD service redis-server start && \
    service postgresql start && \
    sleep 5 && \
    python main.py