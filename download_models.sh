#!/bin/bash

# Create the models directory and its subdirectories
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

# Clone the model repositories
for i in "${!language_codes[@]}"; do
    to_en_repo=${to_en_repos[$i]}
    from_en_repo=${from_en_repos[$i]}
    git clone https://huggingface.co/$to_en_repo ./models/translation/$to_en_repo || { echo "Error cloning repository: $to_en_repo"; exit 1; }
    git clone https://huggingface.co/$from_en_repo ./models/translation/$from_en_repo || { echo "Error cloning repository: $from_en_repo"; exit 1; }
done

# Download the LLM (Vicuna-7B)
git clone https://huggingface.co/TheBloke/Vicuna-7B-v1.5-GGUF ./models/llm/vicuna-7b || { echo "Error downloading Vicuna-7B. Exiting."; exit 1; }

# Download the SentencePiece tokenizer
git clone https://huggingface.co/xlm-roberta-base ./models/embedding/xlm-roberta-base || { echo "Error downloading SentencePiece tokenizer. Exiting."; exit 1; }