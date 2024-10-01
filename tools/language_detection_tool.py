from langchain.tools import BaseTool
from langdetect import detect, lang_detect_exception

class LanguageDetectionTool(BaseTool):
    name = "language_detection"
    description = "Detects the language of the given text."

    def _run(self, text: str) -> str:
        """
        Detect the language of the input text.

        Args:
            text: The text to detect the language of.

        Returns:
            The detected language code (e.g., 'en', 'es', 'fr').
            If detection fails, returns 'en' (English) as the default.
        """
        try:
            detected_language = detect(text)
            return detected_language
        except lang_detect_exception.LangDetectException:
            print(f"Failed to detect language for text: {text}")
            return 'en'  # Default to English if detection fails