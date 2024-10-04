#!/bin/bash

# Check if the 'uploads' directory exists
if [ ! -d "uploads" ]; then
  echo "Error: The 'uploads' directory does not exist. Please create it and add your documents."
  exit 1
fi

# Create a conda environment
conda create -n offlineqa-env python=3.9 || { echo "Error creating conda environment. Exiting."; exit 1; }

# Activate the conda environment
conda activate offlineqa-env || { echo "Error activating conda environment. Exiting."; exit 1; }

# Install dependencies
conda install --file requirements.txt || { echo "Error installing dependencies. Exiting."; exit 1; }
# Install additional dependencies that conda couldn't resolve using pip
pip install -r requirements_pip.txt || { echo "Error installing pip dependencies. Exiting."; exit 1; }

# Download and organize models
mkdir -p models/translation models/llm models/embedding

# Download translation models using Git LFS
git lfs install  # Ensure Git LFS is installed

# Define language codes and model repositories using indexed arrays
language_codes=("es" "fr" "de" "th" "ru" "ar" "pt" "zh")
to_en_repos=(
    "Helsinki-NLP/opus-mt-es-en"
    "Helsinki-NLP/opus-mt-fr-en"
    "Helsinki-NLP/opus-mt-de-en"
    "Helsinki-NLP/opus-mt-th-en"
    "Helsinki-NLP/opus-mt-ru-en"
    "Helsinki-NLP/opus-mt-ar-en"
    "Helsinki-NLP/opus-mt-pt-en"
    "Helsinki-NLP/opus-mt-zh-en"
)
from_en_repos=(
    "Helsinki-NLP/opus-mt-en-es"
    "Helsinki-NLP/opus-mt-en-fr"
    "Helsinki-NLP/opus-mt-en-de"
    "Helsinki-NLP/opus-mt-en-th"
    "Helsinki-NLP/opus-mt-en-ru"
    "Helsinki-NLP/opus-mt-en-ar"
    "Helsinki-NLP/opus-mt-en-pt"
    "Helsinki-NLP/opus-mt-en-zh"
)

# Clone the model repositories, checking if they already exist
for i in "${!language_codes[@]}"; do
    to_en_repo=${to_en_repos[$i]}
    from_en_repo=${from_en_repos[$i]}

    if [ ! -d "models/translation/$to_en_repo-quantized" ]; then
        echo "Cloning '$to_en_repo-quantized'..."
        git clone https://huggingface.co/$to_en_repo ./models/translation/$to_en_repo-quantized || { echo "Error cloning repository: $to_en_repo-quantized"; exit 1; }
    fi

    if [ ! -d "models/translation/$from_en_repo-quantized" ]; then
        echo "Cloning '$from_en_repo-quantized'..."
        git clone https://huggingface.co/$from_en_repo ./models/translation/$from_en_repo-quantized || { echo "Error cloning repository: $from_en_repo-quantized"; exit 1; }
    fi
done

# Download the LLM (Vicuna-7B), checking if it already exists
if [ ! -d "models/llm/vicuna-7b-quantized" ]; then
    echo "Cloning 'Vicuna-7B-quantized'..."
    git clone https://huggingface.co/TheBloke/Vicuna-7B-v1.5-GGUF ./models/llm/vicuna-7b-quantized || { echo "Error downloading Vicuna-7B. Exiting."; exit 1; }
fi

# Download the SentencePiece tokenizer, checking if it already exists
if [ ! -d "models/embedding/xlm-roberta-base" ]; then
    echo "Cloning 'xlm-roberta-base'..."
    git clone https://huggingface.co/xlm-roberta-base ./models/embedding/xlm-roberta-base || { echo "Error downloading SentencePiece tokenizer. Exiting."; exit 1; }
fi

# Generate embeddings for documents
python -c """
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load and preprocess documents from the 'uploads' folder
loader = DirectoryLoader('./uploads', glob='**/*.txt')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Generate embeddings (all-MiniLM-L6-v2 is recommended for Vicuna)
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db = FAISS.from_documents(texts, embeddings)
db.save_local('data/faiss_index')
""" || { echo "Error generating embeddings. Exiting."; exit 1; }

# Download the spaCy language model and enable the coherence pipe
python -m spacy download en_core_web_sm || { echo "Error downloading spaCy model. Exiting."; exit 1; }
python -m spacy_experimental.coref.download en || { echo "Error downloading spaCy coreference data. Exiting."; exit 1; }

# Install and start Redis
conda install -c conda-forge redis || { echo "Error installing Redis. Exiting."; exit 1; }
redis-server --daemonize yes || { echo "Error starting Redis. Exiting."; exit 1; }

# Install PyTorch with dependencies
conda install -c pytorch pytorch torchvision torchaudio || { echo "Error installing PyTorch. Exiting."; exit 1; }

# Install and start PostgreSQL
conda install -c conda-forge postgresql || { echo "Error installing PostgreSQL. Exiting."; exit 1; }
initdb /usr/local/var/postgres || { echo "Error initializing PostgreSQL. Exiting."; exit 1; }
pg_ctl -D /usr/local/var/postgres -l logfile start || { echo "Error starting PostgreSQL. Exiting."; exit 1; }

# Create a database and user in PostgreSQL
psql postgres -c "CREATE DATABASE my_qna_db;" || { echo "Error creating database. Exiting."; exit 1; }
psql postgres -c "CREATE USER my_qna_user WITH ENCRYPTED PASSWORD 'your_password';" || { echo "Error creating user. Exiting."; exit 1; }
psql postgres -c "GRANT ALL PRIVILEGES ON DATABASE my_qna_db TO my_qna_user;" || { echo "Error granting privileges. Exiting."; exit 1; }

echo "Setup complete! You can now run the system using 'python main.py'"