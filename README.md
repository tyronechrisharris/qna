

# Offline Multilingual Question Answering System

This system enables users to submit questions in multiple languages via email and receive accurate, contextually relevant answers in the same language, all while operating entirely offline.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Configuration](#configuration)
    - [Running the System](#running-the-system)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This system is designed to provide a seamless and informative question-answering experience in offline environments where internet connectivity might be limited or unavailable. It leverages a combination of natural language processing, information retrieval, intelligent agents, and email communication to achieve its goals.

## Features

*   **Offline Operation:** Functions entirely without an internet connection, ensuring data privacy and availability
*   **Multilingual Support:** Handles questions and provides answers in multiple languages (Spanish, French, German, Thai, Russian, Arabic, Portuguese, Mandarin)
*   **Contextual Understanding:** Maintains conversation history to provide more relevant and coherent answers to follow-up questions
*   **Self-Correction:**  Employs a retry mechanism to iteratively refine answers, minimizing hallucinations and improving accuracy
*   **Email-Based Interaction:** Offers a convenient and familiar interface for users to submit questions and receive responses via a local mail server
*   **UID Tracking & Database:**  Facilitates tracking of individual interactions, enabling analysis and debugging
*   **Caching:** Enhances performance by storing and reusing previous results
*   **Document Referencing:**  Provides transparency by citing the sources used to generate answers

## System Architecture

The system is built upon a modular architecture, with each component playing a crucial role in the question-answering process.

*   **Email Input Handler:** Fetches new emails from a local Outlook 365 mailbox, checks user registration, extracts questions, generates UIDs, and retrieves context for follow-up questions.
*   **Language Detection:** Identifies the input language using an offline language detection library.
*   **Translation (Conditional):** Translates the question to English if the input language is not English.
*   **Document Retriever:** Retrieves relevant documents from a local embedded document collection using semantic search and metadata filtering.
*   **Contextual Answer Generation:** Generates an answer in English based on the question, retrieved documents, and context.
*   **Self-Corrective Agent:** Evaluates the answer for hallucinations, coherence, and optionally factual accuracy. It can trigger retries if the answer is not acceptable.
*   **Translation (Conditional):** Translates the answer back to the original language if needed.
*   **Email Output Handler:** Constructs and sends the email response, including the answer, references, and UID. It also updates the interaction status in the database.

The system also utilizes a local database to store interaction data and a cache to improve performance by reusing previous results.

![image](/CRAIG_Graph.png)

## Getting Started

### Prerequisites

*   **Python 3.x:** Make sure you have Python 3 installed on your system
*   **Offline Translation Models:** Download the required pre-trained MarianMT models for your supported languages from the Hugging Face Model Hub
*   **Offline LLM:**  Set up and configure an offline large language model (e.g., GPT4All, llama.cpp)
*   **Embedded Document Collection:** Prepare your document collection and embed it using an offline embedding model (e.g., Sentence Transformers)
*   **Local Mail Server:**  Install and configure a local mail server (e.g., Postfix, hMailServer) if you want to retain the email functionality
*   **SQLite3:** Ensure you have SQLite3 installed or choose your preferred database system
*   **Required Libraries:** Install the necessary Python libraries:

    ```bash
    pip install langgraph sentence_transformers spacy langdetect requests
    # Install additional libraries as needed for your specific email and LLM setup
    ```

### Installation

1.  **Clone the repository**
    

    ```bash
    git clone <repository_url> 
    ```
    
2.  **Navigate to the project directory**
    

    ```bash
    cd <project_directory>
    ```
    

### Configuration

1.  **`config.py`**
    *   Update the `language_pairs` dictionary in the `Config` class with the actual paths to your downloaded translation models
    *   Replace the placeholder in `_load_document_collection` with your actual code to load your embedded document collection
    *   Configure the database connection details in `_initialize_database` if you're using a different database system
    *   Update the `email_registry` with the allowed email addresses

2.  **Tool Implementations**
    *   In `email_input_tool.py` and `email_output_tool.py`, replace the placeholders with your actual Outlook 365 integration code using the Microsoft Graph API or a suitable library
    *   In `document_answerer.py`, replace the `OpenAI` placeholder with your offline LLM interface
    *   Customize the `SelfCorrectiveAgent` in `self_corrective_agent.py` with your desired evaluation logic and thresholds

### Running the System

1.  **Execute `main.py`**
    

    ```bash
    python main.py
    ```
    

The system will start running, periodically checking for new emails, processing questions, generating answers, and sending responses

## Customization

*   **Supported Languages:** Add or remove language pairs in the `language_pairs` dictionary in `config.py`
*   **Document Collection:** Update your embedded document collection to expand the system's knowledge base
*   **Answer Generation:**  Experiment with different offline LLM models or fine-tune them for better performance on your specific domain
*   **Self-Correction:**  Customize the evaluation logic and thresholds in the `SelfCorrectiveAgent` to meet your requirements
*   **Email Handling:**  Adapt the Outlook 365 integration code to fit your specific mailbox configuration and email handling needs

## Troubleshooting

*   **Language Detection Failures:** If language detection fails frequently, consider fine-tuning the language detection model or using a different one
*   **Translation Errors:** If translations are inaccurate, try using different pre-trained models or fine-tune them on your specific domain
*   **Hallucinations or Incoherent Answers:** Adjust the self-correction parameters or explore more advanced techniques for evaluating answer quality
*   **Email Issues:**  Check your local mail server configuration and ensure proper authentication and authorization for Outlook 365 access

## Contributing

Contributions to improve and enhance this system are welcome! Please follow these guidelines:

*   Fork the repository
*   Create a new branch for your feature or bug fix
*   Make your changes and commit them with clear and descriptive messages
*   Push your changes to your forked repository
*   Submit a pull request to the main repository

## License

This project is licensed under the [MIT License](LICENSE)

