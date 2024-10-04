# Offline Multilingual Question Answering System

This comprehensive system enables users to interact via a terminal-based chat interface, posing questions in multiple languages and receiving accurate, contextually relevant answers in the same language, all without requiring an internet connection. By leveraging natural language processing, information retrieval, intelligent agents, and offline models, this system prioritizes data privacy and accessibility even in disconnected environments.

## Table of Contents

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
- [Dockerization](#dockerization)

## Introduction

This system is designed to provide a seamless and informative question-answering experience in offline environments where internet connectivity might be limited or unavailable. It employs a terminal-based chat interface for real-time interaction and leverages a local document collection to generate answers.

## Features

- **Offline Operation:** Functions entirely without an internet connection, ensuring data privacy and availability.
- **Multilingual Support:** Handles questions and provides answers in multiple languages (Spanish, French, German, Thai, Russian, Arabic, Portuguese, Mandarin).
- **Contextual Understanding:** Maintains conversation history within chat sessions to provide more relevant and coherent responses to follow-up questions.
- **Self-Correction:** Employs a retry mechanism to iteratively refine answers, minimizing hallucinations and improving accuracy.
- **Terminal-Based Chat Interface:** Offers a user-friendly, real-time chat interface for interaction.
- **UID Tracking & Database:** Assigns unique identifiers to each interaction, facilitating tracking, analysis, and debugging.
- **Caching:** Enhances performance by storing and reusing previous results.
- **Document Referencing:** Provides transparency by citing the sources used to generate answers.
- **Efficient Multilingual Tokenizer:** Utilizes SentencePiece for efficient handling of multiple languages.
- **Offline LLM:** Leverages the Vicuna-7B model for powerful language understanding and answer generation capabilities in an offline setting.
- **Queue-Based Processing:** Handles multiple chat requests concurrently, ensuring fair and efficient processing.
- **User Feedback Mechanism:**  Collects user feedback (thumbs up/thumbs down) to improve the system's performance over time.

![image](/CRAIG_Graph.png)


## System Architecture

The system's modular architecture comprises interconnected components, each fulfilling a specific role in the question-answering process.

*   **`Config`**: Centralizes configuration and setup, including loading models, database connection, cache initialization.
*   **`Tasks`**: Defines the Langgraph tasks and their dependencies, representing the system's workflow.
*   **Tools:** Encapsulate the functionality of different components, facilitating modularity and reusability.
    *   **`LanguageDetectionTool`:** Identifies the input language.
    *   **`Translator`:** Handles translation between supported languages and English.
    *   **`DocumentAnswerer`:** Retrieves relevant documents and generates answers using the offline LLM.
    *   **`SelfCorrectiveAgent`:** Evaluates answers and triggers retries if needed.
    *   **`ChatInputTool`:** Handles user input from the terminal-based chat interface.

## Components

### `Config` (`config.py`)

*   **Purpose:** Centralizes configuration and setup for the system
*   **Functionality:**
    *   Loads offline translation models (MarianMT) for supported language pairs using the SentencePiece tokenizer
    *   Loads the local embedded document collection using the specified embedding model (sentence-transformers/all-MiniLM-L6-v2)
    *   Initializes the database connection and creates the necessary table
    *   Sets up a Redis cache
    *   Loads the offline LLM (Vicuna-7B)

### `Tasks` (`tasks.py`)

*   **Purpose:** Defines the Langgraph tasks and their dependencies, forming the system's workflow
*   **Tasks**
    *   `chat_input_task`: Gets user input from the chat interface and adds it to the request queue
    *   `detect_language_task`: Detects the language of the input question
    *   `translate_to_english_task`: Translates the question to English if needed
    *   `retrieve_documents_task`: Retrieves relevant documents from the local collection
    *   `generate_answer_task`: Generates an answer using the offline LLM and retrieved documents
    *   `self_correct_task`: Evaluates the answer and triggers retries if necessary
    *   `translate_to_user_language_task`: Translates the answer back to the original language if needed
    *   `display_answer_in_chat_task`: Displays the answer in the chat interface and updates the database
*   **Key Considerations:**
    *   The `tool_code` blocks within each task contain the actual logic for performing the task. 
    *   The `args` dictionaries define how data flows between tasks, specifying which outputs from one task are passed as inputs to another

### Tools

#### Language Detection Tool (`language_detection_tool.py`)

*   **Purpose:** Identifies the language of the input text
*   **Functionality:**
    *   Uses the `langdetect` library to detect the language
    *   Handles potential detection failures with a fallback strategy (currently assumes English as the default)

#### Translator (`translator.py`)

*   **Purpose:** Handles translation between supported languages and English
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

### Prerequisites

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
    *   Russian toEnglish: `Helsinki-NLP/opus-mt-ru-en`
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
    *   **Generate embeddings using Sentence Transformers:**

        ```python
        from langchain.document_loaders import TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.vectorstores import FAISS
        from langchain.embeddings import HuggingFaceEmbeddings

        # 1. Load and preprocess documents
        loader = TextLoader('./data/documents/your_document.txt')  # Replace with your actual document path
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)

        # 2. Generate embeddings (all-MiniLM-L6-v2 is recommended for Vicuna)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.from_documents(texts, embeddings)

        # 3. Save the vectorstore (FAISS index) to disk
        db.save_local("data/faiss_index")
        ```

    *   This code snippet demonstrates how to load documents, split them into chunks, generate embeddings using the `all-MiniLM-L6-v2` model (recommended for Vicuna), and store them in a FAISS index.

*   **SQLite3:** 
    *   Ensure you have SQLite3 installed on your system. It's usually included by default in most Python distributions. If not, you can install it using your package manager (e.g., `apt-get install sqlite3` on Debian/Ubuntu or `brew install sqlite3` on macOS).

*   **Required Libraries:** Install the following Python libraries using `pip`:

    ```bash
    pip install langgraph sentence_transformers spacy langdetect requests transformers
    ```

*   **Optional Libraries:** 
    *   If you choose a different caching solution than the basic in-memory cache, install the necessary library for that (e.g., `redis` for Redis).

*   **curses (Windows only):**
    *   If you are using Windows, you'll need to install the `windows-curses` package using `pip`:

        ```bash
        pip install windows-curses
        ```

### Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://code.ornl.gov/6cq/offline-multilingual-question-answering-system](https://code.ornl.gov/6cq/offline-multilingual-question-answering-system)
    ```

2.  **Navigate to the project directory:**

    ```bash
    cd offline-multilingual-question-answering-system
    ```

3.  **Create a document upload folder:**

    *  Create a folder named `uploads` at the root level of the project.

4.  **Add your documents:**

    *  Place your documents (e.g., plain text files) in the `uploads` folder.

5.  **Make the setup script executable:**

    ```bash
    chmod +x MacM3setup.sh
    ```

6.  **Run the setup script:**

    ```bash
    ./MacM3setup.sh  # This will execute the setup script
    ```

    This script will perform the following actions:

    *   **Create and activate a virtual environment:**

        *   It creates a virtual environment named `offlineqa-env` using `python3 -m venv`.
        *   It activates the virtual environment using `source offlineqa-env/bin/activate`.

    *   **Install dependencies:**

        *   It installs the required Python packages listed in `requirements.txt` using `pip install -r requirements.txt`.

    *   **Download and organize models:**

        *   It creates the necessary directories for storing the models (`models/translation`, `models/llm`, `models/embedding`).
        *   It downloads the translation models (MarianMT) for the supported language pairs using the `transformers` library.
        *   It downloads the Vicuna-7B LLM and its tokenizer.
        *   It downloads the SentencePiece tokenizer.

    *   **Prepare your document collection:**

        *   It loads and preprocesses documents from the `uploads` folder using `DirectoryLoader` and `RecursiveCharacterTextSplitter`.
        *   It generates embeddings for the documents using the `all-MiniLM-L6-v2` embedding model (recommended for Vicuna) and stores them in a FAISS index.
        *   It saves the FAISS index to disk (`data/faiss_index`).

    *   **Download the spaCy language model and enable the coherence pipe:**

        *   It downloads the `en_core_web_sm` language model for spaCy using `python -m spacy download en_core_web_sm`.
        *   It downloads the coreference resolution data for spaCy using `python -m spacy_experimental.coref.download en`.

    *   **Set up and start Redis and PostgreSQL:**

        *   It installs Redis and PostgreSQL using Homebrew.
        *   It starts the Redis and PostgreSQL servers.
        *   It creates a database and user in PostgreSQL.

7.  **Make the restart script executable:**

    ```bash
    chmod +x MacM3restart.sh
    ```

8.  **Run the main program:**

    ```bash
    python main.py
    ```

    The system will start running, presenting the terminal-based chat interface for user interaction.

9.  **Restarting the System:**
    *   If you need to restart the system (e.g., after making changes to the code or configuration), you can use the `MacM3restart.sh` script:

        ```bash
        ./MacM3restart.sh
        ```

        This script will restart the Redis and PostgreSQL servers and then re-run the main program.

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

*   **Fine-tune Models:** 
    *   Consider fine-tuning the SentencePiece tokenizer and the translation models on your specific language data to potentially improve performance.
    *   If you have domain-specific data, explore fine-tuning the Vicuna-7B LLM to enhance its accuracy and relevance for your particular use case.
*   **Enhance Self-Correction:**
    *   Investigate and implement more advanced techniques for hallucination detection, coherence assessment, and fact-checking to further improve the quality of generated answers.
*   **Expand Document Collection:**
    *   Continuously update and expand your embedded document collection to cover a wider range of topics and domains, making the system more knowledgeable and versatile.
*   **User Feedback Mechanism:**
    *Develop a more comprehensive system for analyzing and utilizing the collected user feedback to fine-tune models and improve the system's performance.

## Dockerization

This project provides a Dockerfile to simplify the setup and deployment process, allowing you to run the offline multilingual question-answering system within a Docker container.

### Dockerfile

The `Dockerfile` includes instructions to:

*   Use a slim Python 3.9 base image.
*   Install system dependencies (PostgreSQL client, Redis, build tools).
*   Set up the working directory.
*   Install Python dependencies from `requirements.txt`.
*   Download and install the spaCy language model and coreference resolution data.
*   Copy the application code into the container.
*   Start the Redis and PostgreSQL servers.
*   Execute the `main.py` script to start the question-answering system.

### Building the Docker Image

1.  Ensure you have Docker installed on your system. You can download it from the official Docker website: [https://www.docker.com/](https://www.docker.com/)

2.  Open a terminal or command prompt and navigate to the project's root directory.

3.  Build the Docker image using the following command:

    ```bash
    docker build -t offline-qna-system .
    ```

    This will create a Docker image named `offline-qna-system`.

### Running the Docker Container

1.  Run the Docker container in interactive mode:

    ```bash
    docker run -it offline-qna-system
    ```

    This will start the container, and you should see the chat interface in the terminal.

2.  **Set up the database:**

    *   In a separate terminal window, access the PostgreSQL shell within the running container:

        ```bash
        docker exec -it <container_id> sudo -u postgres psql  # Replace <container_id> with the actual container ID
        ```

    *   Create a database and user (replace with your desired names):

        ```sql
        CREATE DATABASE my_qna_db;
        CREATE USER my_qna_user WITH ENCRYPTED PASSWORD 'your_password';
        GRANT ALL PRIVILEGES ON DATABASE my_qna_db TO my_qna_user;
        \q  # Exit the shell
        ```

3.  **Interact with the system:**

    *   Use the chat interface in the container's terminal to ask questions and receive answers.

**Important Considerations**

*   **Data Persistence:** By default, any data stored in the container (database, cache) will be lost when the container is stopped. To persist data, you can use Docker volumes to mount directories from your host machine into the container.
*   **Resource Allocation:** Adjust the resource allocation (CPU, memory) for the container based on your hardware capabilities and the requirements of the models and services.
*   **Security:**  In a production environment, it's crucial to configure PostgreSQL and Redis with appropriate security measures.
*   **Port Mapping:** If you want to access Redis or PostgreSQL from outside the container, you'll need to map the container's ports to your host machine's ports using the `-p` option when running the container (e.g., `docker run -it -p 6379:6379 offline-qna-system`).

**Example with Volume Mounting and Port Mapping**

```bash
docker run -it \
       -v /path/to/your/data:/app/data \  # Mount data directory for persistence
       -p 6379:6379 \  # Map Redis port
       offline-qna-system
```       

This command mounts the /path/to/your/data directory from your host machine to the /app/data directory inside the container, ensuring that any data stored in the database or cache is persisted even after the container stops. It also maps the container's Redis port (6379) to the same port on the host machine, allowing you to access Redis from outside the container.

