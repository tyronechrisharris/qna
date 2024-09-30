import sqlite3
from transformers import MarianMTModel, MarianTokenizer
from langchain.embeddings import OpenAIEmbeddings  # Replace if using a different embedding model
from langchain.vectorstores import FAISS         # Replace if using a different vectorstore

class Config:
    def __init__(self):
        # 1. Load Offline Translation Models
        self.tokenizers, self.models = self._load_translation_models()

        # 2. Load Embedded Document Collection
        self.vectorstore = self._load_document_collection()

        # 3. Initialize Database Connection
        self.conn, self.cursor = self._initialize_database()

        # 4. Initialize Cache
        self.cache = {}  # Simple in-memory cache, replace with your preferred implementation

        # 5. Email Registry
        self.email_registry = ["user1@example.com", "user2@example.com"]  # Replace with your actual allowed emails

    def _load_translation_models(self):
        # Define language codes and model names (replace with your actual models)
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

        # Load tokenizers and models
        tokenizers = {}
        models = {}
        for lang_code, (to_en_model_name, from_en_model_name) in language_pairs.items():
            tokenizers[lang_code] = {
                'to_en': MarianTokenizer.from_pretrained(to_en_model_name),
                'from_en': MarianTokenizer.from_pretrained(from_en_model_name)
            }
            models[lang_code] = {
                'to_en': MarianMTModel.from_pretrained(to_en_model_name),
                'from_en': MarianMTModel.from_pretrained(from_en_model_name)
            }

        return tokenizers, models

    def _load_document_collection(self):
        # Load your local embedded document collection (replace with your actual implementation)
        return FAISS.load_local("my_faiss_index", OpenAIEmbeddings())

    def _initialize_database(self):
        conn = sqlite3.connect('email_interactions.db')
        cursor = conn.cursor()

        # Create the email_interactions table (run this once)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS email_interactions (
                uid TEXT PRIMARY KEY,
                sender_email TEXT,
                timestamp DATETIME,
                question TEXT,
                answer TEXT,
                status TEXT,
                other_metadata TEXT
            )
        ''')
        conn.commit()

        return conn, cursor