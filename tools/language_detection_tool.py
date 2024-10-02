import logging

from langchain.tools import BaseTool
from langdetect import detect, lang_detect_exception

class LanguageDetectionTool(BaseTool):
    """
    Tool for detecting the language of a given text using the 'langdetect' library.
    Includes error handling and logging.
    """

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
        except lang_detect_exception.LangDetectException as e:
            logging.error(f"Language detection failed for text: '{text}' - Error: {e}")
            # You can add a user notification here if needed
            return 'en'  # Default to English if detection fails
        except Exception as e:
            logging.error(f"An unexpected error occurred during language detection: {e}")
            # You can add a user notification here if needed
            return 'en'  # Default to English if an unexpected error occurs