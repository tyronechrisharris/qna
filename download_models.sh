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

# Clone the model repositories, checking if they already exist
for i in "${!language_codes[@]}"; do
    to_en_repo=${to_en_repos[$i]}
    from_en_repo=${from_en_repos[$i]}

    if [ ! -d "models/translation/$to_en_repo-quantized" ]; then
        echo "Cloning '$to_en_repo-quantized'..."
        git clone https://huggingface.co/$to_en_repo ./models/translation/$to_en_repo-quantized || { echo "Error cloning repository: $to_en_repo-quantized"; exit 1; }
    else
        echo "Updating '$to_en_repo-quantized'..."
        (cd "models/translation/$to_en_repo-quantized" && git pull) || { echo "Error updating repository: $to_en_repo-quantized"; exit 1; }
    fi

    if [ ! -d "models/translation/$from_en_repo-quantized" ]; then
        echo "Cloning '$from_en_repo-quantized'..."
        git clone https://huggingface.co/$from_en_repo ./models/translation/$from_en_repo-quantized || { echo "Error cloning repository: $from_en_repo-quantized"; exit 1; }
    else
        echo "Updating '$from_en_repo-quantized'..."
        (cd "models/translation/$from_en_repo-quantized" && git pull) || { echo "Error updating repository: $from_en_repo-quantized"; exit 1; }
    fi
done

# Download the LLM (Vicuna-7B), checking if it already exists
if [ ! -d "models/llm/vicuna-7b-quantized" ]; then
    echo "Cloning 'Vicuna-7B-quantized'..."
    git clone https://huggingface.co/TheBloke/Vicuna-7B-v1.5-GGUF ./models/llm/vicuna-7b-quantized || { echo "Error downloading Vicuna-7B. Exiting."; exit 1; }
else
    echo "Updating 'Vicuna-7B-quantized'..."
    (cd "models/llm/vicuna-7b-quantized" && git pull) || { echo "Error updating Vicuna-7B."; exit 1; }
fi

# Download the SentencePiece tokenizer, checking if it already exists
if [ ! -d "models/embedding/xlm-roberta-base" ]; then
    echo "Cloning 'xlm-roberta-base'..."
    git clone https://huggingface.co/xlm-roberta-base ./models/embedding/xlm-roberta-base || { echo "Error downloading SentencePiece tokenizer. Exiting."; exit 1; }
else
    echo "Updating 'xlm-roberta-base'..."
    (cd "models/embedding/xlm-roberta-base" && git pull) || { echo "Error updating xlm-roberta-base."; exit 1; }
fi