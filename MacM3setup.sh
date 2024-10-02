#!/bin/bash

# Check if the 'uploads' directory exists
if [ ! -d "uploads" ]; then
  echo "Error: The 'uploads' directory does not exist. Please create it and add your documents."
  exit 1
fi

# Create a virtual environment
python3 -m venv offlineqa-env || { echo "Error creating virtual environment. Exiting."; exit 1; }

# Activate the virtual environment
source offlineqa-env/bin/activate || { echo "Error activating virtual environment. Exiting."; exit 1; }

# Install dependencies
pip install -r requirements.txt || { echo "Error installing dependencies. Exiting."; exit 1; }

# Download and organize models
mkdir -p models/translation models/llm models/embedding

# Download translation models
python -c """
from transformers import MarianMTModel, MarianTokenizer

language_pairs = {
    'es': ('Helsinki-NLP/opus-mt-es-en', 'Helsinki-NLP/opus-mt-en-es'),
    'fr': ('Helsinki-NLP/opus-mt-fr-en', 'Helsinki-NLP/opus-mt-en-fr'),
    'de': ('Helsinki-NLP/opus-mt-de-en', 'Helsinki-NLP/opus-mt-en-de'),
    'th': ('Helsinki-NLP/opus-mt-th-en', 'Helsinki-NLP/opus-mt-en-th'),
    'ru': ('Helsinki-NLP/opus-mt-ru-en', 'Helsinki-NLP/opus-mt-en-ru'),
    'ar': ('Helsinki-NLP/opus-mt-ar-en', 'Helsinki-NLP/opus-mt-en-ar'),
    'pt': ('Helsinki-NLP/opus-mt-pt-en', 'Helsinki-NLP/opus-mt-en-pt'),
    'zh': ('Helsinki-NLP/opus-mt-zh-en', 'Helsinki-NLP/opus-mt-en-zh'),
}

for _, (to_en_model_name, from_en_model_name) in language_pairs.items():
    MarianTokenizer.from_pretrained(to_en_model_name, cache_dir='./models/translation')
    MarianMTModel.from_pretrained(to_en_model_name, cache_dir='./models/translation')
    MarianTokenizer.from_pretrained(from_en_model_name, cache_dir='./models/translation')
    MarianMTModel.from_pretrained(from_en_model_name, cache_dir='./models/translation')
""" || { echo "Error downloading translation models. Exiting."; exit 1; }

# Download the LLM (Vicuna-7B)
python -c """
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('TheBloke/Vicuna-7B-v1.5-GGUF', cache_dir='./models/llm')
model = AutoModelForCausalLM.from_pretrained('TheBloke/Vicuna-7B-v1.5-GGUF', cache_dir='./models/llm')
""" || { echo "Error downloading Vicuna-7B. Exiting."; exit 1; }

# Download the SentencePiece tokenizer
python -c """
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', cache_dir='./models/embedding')
""" || { echo "Error downloading SentencePiece tokenizer. Exiting."; exit 1; }

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
brew install redis || { echo "Error installing Redis. Exiting."; exit 1; }
brew services start redis || { echo "Error starting Redis. Exiting."; exit 1; }

# Install and start PostgreSQL
brew install postgresql || { echo "Error installing PostgreSQL. Exiting."; exit 1; }
brew services start postgresql || { echo "Error starting PostgreSQL. Exiting."; exit 1; }

# Create a database and user in PostgreSQL
psql postgres -c "CREATE DATABASE my_qna_db;" || { echo "Error creating database. Exiting."; exit 1; }
psql postgres -c "CREATE USER my_qna_user WITH ENCRYPTED PASSWORD 'your_password';" || { echo "Error creating user. Exiting."; exit 1; }
psql postgres -c "GRANT ALL PRIVILEGES ON DATABASE my_qna_db TO my_qna_user;" || { echo "Error granting privileges. Exiting."; exit 1; }

echo "Setup complete! You can now run the system using 'python main.py'"