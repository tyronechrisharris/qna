from langchain.tools import BaseTool
from langdetect import detect

class Translator(BaseTool):
    """
    Tool for translating text between various languages and English.
    Leverages the SentencePiece tokenizer and pre-trained MarianMT models.
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
        cached_translation = self.cache.get(cache_key)
        if cached_translation:
            return cached_translation

        # 2. Detect source language if not provided
        source_language = detect(text) if target_language == 'en' else 'en'

        # 3. Perform translation
        try:
            model = self.models[source_language][f'to_{target_language}']
        except KeyError:
            # Handle unsupported language pairs
            return f"Translation from {source_language} to {target_language} is not supported."

        # Use the SentencePiece tokenizer
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        outputs = model.generate(input_ids)
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 4. Store in cache
        self.cache.set(cache_key, translation)

        return translation

    def _arun(self, text):
        """Asynchronous translation is not currently supported."""
        raise NotImplementedError("Translator does not support async")