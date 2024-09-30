from langchain.tools import BaseTool
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect

class Translator(BaseTool):
    name = "translator"
    description = "Translate text between supported languages and English."

    def __init__(self, cache, tokenizers, models):
        super().__init__()
        self.cache = cache
        self.tokenizers = tokenizers
        self.models = models

    def _run(self, text, target_language='en'):
        # 1. Check cache for existing translation
        cache_key = (text, target_language)
        cached_translation = self.cache.get(cache_key)
        if cached_translation:
            return cached_translation

        # 2. Detect source language if not provided
        source_language = detect(text) if target_language == 'en' else 'en'

        # 3. Perform translation
        try:
            tokenizer = self.tokenizers[source_language][f'to_{target_language}']
            model = self.models[source_language][f'to_{target_language}']
        except KeyError:
            # Handle unsupported language pairs
            return f"Translation from {source_language} to {target_language} is not supported."

        input_ids = tokenizer.encode(text, return_tensors="pt")
        outputs = model.generate(input_ids)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 4. Store in cache
        self.cache.set(cache_key, translation)

        return translation

    def _arun(self, text):
        raise NotImplementedError("Translator does not support async")