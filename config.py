import logging
import sqlite3

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from redis import Redis  # Import Redis library

# Set up logging
logging.basicConfig(filename='qna_system.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(filename)s - %(message)s')


class Config:
    def __init__(self):
        # 1. Load Offline Translation Models
        try:
            self.tokenizer, self.models = self._load_translation_models()
        except Exception as e:
            logging.error(f"Error loading translation models: {e}")
            raise

        # 2. Load Embedded Document Collection
        try:
            self.vectorstore = self._load_document_collection()
        except Exception as e:
            logging.error(f"Error loading document collection: {e}")
            raise

        # 3. Initialize Database Connection
        try:
            self.conn, self.cursor = self._initialize_database()
        except Exception as e:
            logging.error(f"Error initializing database: {e}")
            raise

        # 4. Initialize Redis Cache
        try:
            self.cache = Redis(host='localhost', port=6379, db=0)  # Configure Redis connection
        except Exception as e:
            logging.error(f"Error connecting to Redis: {e}")
            raise

        # 5. Load LLM 
        try:
            self.llm = self._load_llm()
        except Exception as e:
            logging.error(f"Error loading LLM: {e}")
            raise

    def _load_translation_models(self):
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

        # Load the multilingual SentencePiece tokenizer
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base") 

        # Load models
        models = {}
        for lang_code, (to_en_model_name, from_en_model_name) in language_pairs.items():
            models[lang_code] = {
                'to_en': MarianMTModel.from_pretrained(to_en_model_name),
                'from_en': MarianMTModel.from_pretrained(from_en_model_name)
            }

        return tokenizer, models

    def _load_document_collection(self):
        # Load your local embedded document collection 
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.load_local("data/faiss_index", embeddings)

    def _initialize_database(self):
        conn = sqlite3.connect('chat_interactions.db')  # Changed database name to 'chat_interactions.db'
        cursor = conn.cursor()

        # Create the chat_interactions table 
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_interactions (
                uid TEXT PRIMARY KEY,
                timestamp DATETIME,
                question TEXT,
                answer TEXT,
                status TEXT,
                other_metadata TEXT
            )
        ''')
        conn.commit()

        return conn, cursor

    def _load_llm(self):
        tokenizer = AutoTokenizer.from_pretrained("TheBloke/Vicuna-7B-v1.5-GGUF")
        model = AutoModelForCausalLM.from_pretrained("TheBloke/Vicuna-7B-v1.5-GGUF")
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer,
            max_new_tokens=256, 
            temperature=0.7,
            top_p=0.95, 
            repetition_penalty=1.15,
            top_k=40
        ) 
        return HuggingFacePipeline(pipeline=pipe)