import logging

from langchain.tools import BaseTool
from langdetect import detect, lang_detect_exception

class Translator(BaseTool):
    """
    Tool for translating text between various languages and English.
    Leverages the SentencePiece tokenizer and pre-trained MarianMT models.
    Includes robust error handling and logging.
    """

    name = "translator"
    description = "Translate text between supported languages and English."

    def __init__(self, cache, tokenizer, models):
        """Initialize the Translator with cache, tokenizer, and models.

        Args:
            cache: The cache object to store and retrieve translations.
            tokenizer: The SentencePiece tokenizer for handling multiple languages.
            models: A dictionary containing language-specific translation models.
        """        
        super().__init__()
        self.cache = cache
        self.tokenizer = tokenizer
        self.models = models

    def _run(self, text, target_language='en'):
        """Translate the given text to the target language.

        Args:
            text: The text to be translated.
            target_language: The target language code (default is 'en' for English).

        Returns:
            The translated text or an error message if the translation is not supported.
        """

        # 1. Check cache for existing translation
        cache_key = (text, target_language)
        try:
            cached_translation = self.cache.get(cache_key)
            if cached_translation:
                return cached_translation.decode('utf-8')  # Decode from bytes to string
        except Exception as e:
            logging.error(f"Error accessing cache in Translator: {e}")
            # ... (potential fallback strategy or user notification)

        # 2. Detect source language if not provided
        try:
            source_language = detect(text) if target_language == 'en' else 'en'
        except lang_detect_exception.LangDetectException as e:
            logging.error(f"Language detection failed for text: '{text}' - Error: {e}")
            return "Error: Could not detect the source language for translation."
        except Exception as e:
            logging.error(f"An unexpected error occurred during language detection: {e}")
            return "Error: Could not detect the source language for translation."

        # 3. Perform translation
        try:
            model = self.models[source_language][f'to_{target_language}']

            # Use the SentencePiece tokenizer
            input_ids = self.tokenizer.encode(text, return_tensors="pt")
            outputs = model.generate(input_ids)
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        except KeyError as e:
            logging.error(f"Unsupported language pair for translation: {e}")
            return f"Error: Translation between the detected language and {target_language} is not supported."
        except Exception as e:
            logging.error(f"An unexpected error occurred during translation: {e}")
            return "Error: Could not translate the text."

        # 4. Store in cache (store as bytes)
        try:
            self.cache.set(cache_key, translation.encode('utf-8'))
        except Exception as e:
            logging.error(f"Error storing translation in cache: {e}")
            # ... (potential fallback strategy or user notification)

        return translation

    def _arun(self, text):
        """Asynchronous translation is not currently supported."""
        raise NotImplementedError("Translator does not support async")