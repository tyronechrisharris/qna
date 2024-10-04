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

# Download and install spaCy language model and coreference data
RUN python -m spacy download en_core_web_sm
RUN python -m spacy_experimental.coref.download en

# Copy the application code
COPY . .

# Expose ports for Redis and PostgreSQL (optional, if you want to access them externally)
# EXPOSE 6379  # Redis default port
# EXPOSE 5432  # PostgreSQL default port

# Define commands to start Redis and PostgreSQL servers, and then run the main program
CMD service redis-server start && \
    service postgresql start && \
    sleep 5 && \
    python -c "from langchain.document_loaders import DirectoryLoader; from langchain.text_splitter import RecursiveCharacterTextSplitter; from langchain.vectorstores import FAISS; from langchain.embeddings import HuggingFaceEmbeddings; loader = DirectoryLoader('/uploads', glob='**/*.txt'); documents = loader.load(); text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0); texts = text_splitter.split_documents(documents); embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'); db = FAISS.from_documents(texts, embeddings); db.save_local('data/faiss_index')" && \
    python main.py