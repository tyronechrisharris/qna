## Offline Multilingual Question Answering System

This comprehensive system enables users to interact via a terminal-based chat interface, posing questions in multiple languages and receiving accurate, contextually relevant answers, all without requiring an internet connection. By leveraging natural language processing, information retrieval, intelligent agents, and offline models, this system prioritizes data privacy and accessibility even in disconnected environments.

### Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Components](#components)
    - [Config (`config.py`)](#config-configpy)
    - [Tasks (`tasks.py`)](#tasks-taskspy)
    - [Tools](#tools)
        - [Language Detection Tool (`language_detection_tool.py`)](#language-detection-tool-language_detection_toolpy)
        - [Translator (`translator.py`)](#translator-translatorpy)
        - [Document Answerer (`document_answerer.py`)](#document-answerer-document_answererpy)
        - [Self-Corrective Agent (`self_corrective_agent.py`)](#self-corrective-agent-self_corrective_agentpy)
        - [Chat Input Tool (`chat_input_tool.py`)](#chat-input-tool-chat_input_toolpy)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Configuration](#configuration)
    - [Running the System](#running-the-system)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [TODO](#todo)

## Introduction

This system is designed to provide a seamless and informative question-answering experience in offline environments where internet connectivity might be limited or unavailable. It employs a terminal-based chat interface for real-time interaction and leverages a local document collection to generate answers.

## Features

- **Offline Operation:**  Functions entirely without an internet connection, ensuring data privacy and availability.
- **Multilingual Support:** Handles questions and provides answers in multiple languages (Spanish, French, German, Thai, Russian, Arabic, Portuguese, Mandarin).
- **Contextual Understanding:** Maintains conversation history within chat sessions to provide more relevant and coherent responses to follow-up questions.
- **Self-Correction:**  Employs a retry mechanism to iteratively refine answers, minimizing hallucinations and improving accuracy.
- **Terminal-Based Chat Interface:**  Offers a user-friendly, real-time chat interface for interaction.
- **UID Tracking & Database:**  Assigns unique identifiers to each interaction, facilitating tracking, analysis, and debugging.
- **Caching:**  Enhances performance by storing and reusing previous results.
- **Document Referencing:**  Provides transparency by citing the sources used to generate answers.
- **Efficient Multilingual Tokenizer:** Utilizes SentencePiece for efficient handling of multiple languages.
- **Offline LLM:**  Leverages the Vicuna-7B model for powerful language understanding and answer generation capabilities in an offline setting.
- **Queue-Based Processing:**  Handles multiple chat requests concurrently, ensuring fair and efficient processing.

![image](/CRAIG_Graph-2024-10-01-035038.svg)

## System Architecture

The system is built upon a modular architecture, orchestrated using Langgraph, a declarative workflow management framework. It consists of several key components, each responsible for a specific task in the question-answering process:

*   **`Config`**: Centralizes configuration and setup, including loading models, database connection, cache initialization.
*   **`Tasks`**:  Defines the Langgraph tasks and their dependencies, representing the system's workflow.
*   **Tools:** Encapsulate the functionality of different components, facilitating modularity and reusability.
    *   **`LanguageDetectionTool`:** Identifies the input language.
    *   **`Translator`:** Handles translation between supported languages and English.
    *   **`DocumentAnswerer`:** Retrieves relevant documents and generates answers using the offline LLM.
    *   **`SelfCorrectiveAgent`:** Evaluates answers and triggers retries if needed.
    *   **`ChatInputTool`:**  Handles user input from the terminal-based chat interface.

## Components

### `Config` (`config.py`)

*   **Purpose:** Centralizes configuration and setup for the system
*   **Functionality:**
    *   Loads offline translation models (MarianMT) for supported language pairs using the SentencePiece tokenizer
    *   Loads the local embedded document collection using the specified embedding model (sentence-transformers/all-MiniLM-L6-v2)
    *   Initializes the database connection and creates the necessary table
    *   Sets up an in-memory cache (can be replaced with a more robust solution)
    *   Loads the offline LLM (Vicuna-7B)

### `Tasks` (`tasks.py`)

*   **Purpose:** Defines the Langgraph tasks and their dependencies, forming the system's workflow
*   **Tasks**
    *   `fetch_new_emails_task`:  (Placeholder for future email integration)
    *   `check_registry_task`:  (Placeholder for future email integration)
    *   `chat_input_task`:  Gets user input from the chat interface and adds it to the request queue
    *   `detect_language_task`:  Detects the language of the input question
    *   `translate_to_english_task`: Translates the question to English if needed
    *   `retrieve_documents_task`: Retrieves relevant documents from the local collection
    *   `generate_answer_task`: Generates an answer using the offline LLM and retrieved documents
    *   `self_correct_task`:  Evaluates the answer and triggers retries if necessary
    *   `translate_to_user_language_task`: Translates the answer back to the original language if needed
    *   `display_answer_in_chat_task`: Displays the answer in the chat interface and updates the database
*   **Key Considerations:**
    *   The `tool_code` blocks within each task contain the actual logic for performing the task. You'll need to fill in the placeholders with your specific implementations
    *   The `args` dictionaries define how data flows between tasks, specifying which outputs from one task are passed as inputs to another

### Tools

#### Language Detection Tool (`language_detection_tool.py`)

*   **Purpose:** Identifies the language of the input text
*   **Functionality:**
    *   Uses the `langdetect` library to detect the language
    *   Handles potential detection failures with a fallback strategy (currently assumes English as the default)

#### Translator (`translator.py`)

*   **Purpose:**  Handles translation between supported languages and English
*   **Functionality:**
    *   Uses the SentencePiece tokenizer and pre-trained MarianMT models loaded in `config.py`
    *   Caches translations for efficiency
    *   Detects the source language if the target language is English
    *   Handles unsupported language pairs with an error message

#### Document Answerer (`document_answerer.py`)

*   **Purpose:** Retrieves relevant documents and generates answers
*   **Functionality:**
    *   Uses the local vectorstore to retrieve documents semantically similar to the query
    *   Incorporates context from previous interactions for follow-up questions
    *   Generates answers using the offline LLM
    *   Caches answers for efficiency
    *   Includes references to the documents used in the response

#### Self-Corrective Agent (`self_corrective_agent.py`)

*   **Purpose:** Evaluates the quality of generated answers
*   **Functionality:**
    *   Checks for hallucinations, coherence, and sense
    *   Optionally, performs factual accuracy checks (requires additional implementation)
    *   Triggers retries with feedback to the Document Retriever if the answer is not acceptable (up to 3 retries)
    *   If max retries are reached, passes the answer along with identified problems

#### Chat Input Tool (`chat_input_tool.py`)

*   **Purpose:** Handles user input from the terminal-based chat interface
*   **Functionality:**
    *   Uses the `curses` library to create an interactive chat window in the terminal
    *   Gets user input, generates UIDs for new chat sessions, and retrieves context for follow-up questions
    *   Adds chat requests to the queue for processing

## Getting Started

## Prerequisites

*   **Python 3.x:**  Ensure you have Python 3.x installed on your system. You can download it from the official Python website: [https://www.python.org/](https://www.python.org/)

*   **Offline Translation Models:**  Download the following pre-trained MarianMT models for the supported languages from the Hugging Face Model Hub:

    *   Spanish to English: `Helsinki-NLP/opus-mt-es-en`
    *   English to Spanish: `Helsinki-NLP/opus-mt-en-es`
    *   French to English: `Helsinki-NLP/opus-mt-fr-en`
    *   English to French: `Helsinki-NLP/opus-mt-en-fr`
    *   German to English: `Helsinki-NLP/opus-mt-de-en`
    *   English to German: `Helsinki-NLP/opus-mt-en-de`
    *   Thai to English: `Helsinki-NLP/opus-mt-th-en`
    *   English to Thai: `Helsinki-NLP/opus-mt-en-th`
    *   Russian to English: `Helsinki-NLP/opus-mt-ru-en`
    *   English to Russian: `Helsinki-NLP/opus-mt-en-ru`
    *   Arabic to English: `Helsinki-NLP/opus-mt-ar-en`
    *   English to Arabic: `Helsinki-NLP/opus-mt-en-ar`
    *   Portuguese to English: `Helsinki-NLP/opus-mt-pt-en`
    *   English to Portuguese: `Helsinki-NLP/opus-mt-en-pt`
    *   Mandarin to English: `Helsinki-NLP/opus-mt-zh-en`
    *   English to Mandarin: `Helsinki-NLP/opus-mt-en-zh`

*   **Offline LLM:** 
    *   Download and set up the **Vicuna-7B** offline large language model. You can find instructions and resources on the project's GitHub page or other relevant sources.

*   **Embedded Document Collection:** 
    *   Prepare your document collection in a suitable format (e.g., plain text files).
    *   Embed the documents using an offline embedding model like **Sentence Transformers**. 
    *   Store the embeddings in a local vector database (e.g., FAISS) for efficient retrieval.

*   **SQLite3:** 
    *   Ensure you have SQLite3 installed on your system. It's usually included by default in most Python distributions. If not, you can install it using your package manager (e.g., `apt-get install sqlite3` on Debian/Ubuntu or `brew install sqlite3` on macOS).

*   **Required Libraries:** Install the following Python libraries using `pip`:

    ```bash
    pip install langgraph sentence_transformers spacy langdetect requests transformers
    ```

*   **Optional Libraries:** 
    *   If you plan to implement the email functionality in the future, you'll also need to install libraries for interacting with the Outlook 365 API (e.g., `requests_oauthlib` and `microsoft-graph`).
    *   If you choose a different caching solution than the basic in-memory cache, install the necessary library for that (e.g., `redis` for Redis).

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://code.ornl.gov/6cq/offline-multilingual-question-answering-system
    ```

2.  **Navigate to the project directory:**

    ```bash
    cd offline-multilingual-question-answering-system
    ```

3.  **Create a virtual environment:**

    ```bash
    python -m venv myenv  # Replace 'myenv' with your preferred environment name
    ```

4.  **Activate the virtual environment:**

    *   **On Windows:**

        ```bash
        myenv\Scripts\activate
        ```

    *   **On macOS/Linux:**

        ```bash
        source myenv/bin/activate
        ```

5.  **Install dependencies using the provided `requirements.txt` file:**

    ```bash
    pip install -r requirements.txt
    ```

    This will install all the necessary Python packages listed in `requirements.txt` within the virtual environment, ensuring that your project has its own isolated set of dependencies. 

6.  **Download and organize models:**

    *   **Create the `models` directory and its subdirectories:**

        ```bash
        mkdir -p models/translation models/llm models/embedding
        ```

    *   **Download translation models:**

        ```python
        from transformers import MarianMTModel, MarianTokenizer

        # Define language codes and model names 
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

        # Download models
        for _, (to_en_model_name, from_en_model_name) in language_pairs.items():
            MarianTokenizer.from_pretrained(to_en_model_name, cache_dir="./models/translation")
            MarianMTModel.from_pretrained(to_en_model_name, cache_dir="./models/translation")
            MarianTokenizer.from_pretrained(from_en_model_name, cache_dir="./models/translation")
            MarianMTModel.from_pretrained(from_en_model_name, cache_dir="./models/translation")
        ```


*   **Download the LLM (Vicuna-7B):**

    *   **Ensure you have enough disk space:** The Vicuna-7B model is quite large (around 13GB). Make sure you have sufficient disk space available before downloading
    *   **Use the `transformers` library to download:** 

        ```python
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained("TheBloke/Vicuna-7B-v1.5-GGUF", cache_dir="./models/llm")
        model = AutoModelForCausalLM.from_pretrained("TheBloke/Vicuna-7B-v1.5-GGUF", cache_dir="./models/llm")
        ```

        This code will download both the tokenizer and the model weights to the `./models/llm` directory

    *   **Consider using a quantized version:** If you're running the system on a CPU or have limited memory, you might want to explore using a quantized version of Vicuna-7B, which can significantly reduce its memory footprint and improve inference speed. Refer to the Vicuna-7B model documentation for instructions on how to obtain and use a quantized version.

*   **Download the SentencePiece tokenizer:**

    ```python
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", cache_dir="./models/embedding")
    ```

    This code will download the SentencePiece tokenizer to the `./models/embedding` directory.

*   **Prepare your document collection:**

    *   Organize your documents in a suitable format (e.g., plain text files) within the `data/documents` directory.
    *   Use an offline embedding model (e.g., Sentence Transformers) to generate embeddings for your documents and store them in a local vector database (e.g., FAISS) for efficient retrieval. You can refer to the Langchain documentation or other resources for guidance on how to perform document embedding and indexing.




### Configuration

1.  **`config.py`**
    *   Update the `language_pairs` dictionary in the `Config` class with the actual paths to your downloaded translation models.
    *   Replace the placeholder in `_load_document_collection` with your actual code to load your embedded document collection.
    *   Configure the database connection details in `_initialize_database` if you're using a different database system.
    *   Ensure you have the SentencePiece tokenizer downloaded and specify its path in `_load_translation_models`.

2.  **Tool Implementations**
    *   In `document_answerer.py`, replace the `OpenAI` placeholder with your actual offline LLM interface.
    *   Customize the `SelfCorrectiveAgent` in `self_corrective_agent.py` with your desired evaluation logic and thresholds.
    *   Implement the chat interface logic in `chat_input_tool.py` and the `display_answer_in_chat_task` in `tasks.py` using the `curses` library or a similar approach.

### Running the System

1.  **Execute `main.py`**

    ```bash
    python main.py
    ```

    The system will start running, presenting the terminal-based chat interface for user interaction.

## Customization

*   **Supported Languages:** Add or remove language pairs in the `language_pairs` dictionary in `config.py`.
*   **Document Collection:** Update your embedded document collection to expand the system's knowledge base.
*   **Answer Generation:** Experiment with different offline LLM models or fine-tune them for better performance on your specific domain.
*   **Self-Correction:** Adjust the evaluation logic and thresholds in the `SelfCorrectiveAgent` to meet your requirements.
*   **Chat Interface:** Enhance the terminal-based chat interface using `curses` or explore other chat interface libraries for a richer user experience.

## Troubleshooting

*   **Language Detection Failures:** If language detection fails frequently, consider fine-tuning the language detection model or using a different one.
*   **Translation Errors:** If translations are inaccurate, try using different pre-trained models or fine-tune them on your specific domain.
*   **Hallucinations or Incoherent Answers:** Adjust the self-correction parameters or explore more advanced techniques for evaluating answer quality.

## Contributing

Contributions to improve and enhance this system are welcome! Please follow these guidelines:

*   Fork the repository
*   Create a new branch for your feature or bug fix
*   Make your changes and commit them with clear and descriptive messages
*   Push your changes to your forked repository
*   Submit a pull request to the main repository

## License

This project is licensed under the [MIT License](LICENSE)

## TODO

*   **Implement Outlook 365 Integration (Optional):** 
    *   If you want to add email functionality in the future, replace the placeholders in `email_input_tool.py` and `email_output_tool.py` with your actual Outlook 365 integration code using the Microsoft Graph API or a suitable library.
*   **Fine-tune Models:** 
    *   Consider fine-tuning the SentencePiece tokenizer and the translation models on your specific language data to potentially improve performance.
    *   If you have domain-specific data, explore fine-tuning the Vicuna-7B LLM to enhance its accuracy and relevance for your particular use case.
*   **Enhance Self-Correction:**
    *   Investigate and implement more advanced techniques for hallucination detection, coherence assessment, and fact-checking to further improve the quality of generated answers.
*   **Implement Robust Caching:** 
    *   Replace the simple in-memory cache with a more production-ready solution like Redis or Memcached, especially if you anticipate high volumes of interactions.
    *   Implement proper cache expiration and invalidation strategies to manage memory usage and ensure answer freshness.
*   **Expand Document Collection:**
    *   Continuously update and expand your embedded document collection to cover a wider range of topics and domains, making the system more knowledgeable and versatile.
*   **User Feedback Mechanism:**
    *   Incorporate a mechanism to collect user feedback on the quality and relevance of answers. This feedback can be used to further fine-tune models and improve the system's overall performance.


