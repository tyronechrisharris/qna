from langchain.tools import BaseTool
from langdetect import detect, lang_detect_exception

class LanguageDetectionTool(BaseTool):
    name = "language_detection"
    description = "Detects the language of the given text."

    def _run(self, text):
        try:
            # Attempt language detection using langdetect
            detected_language = detect(text)
            return detected_language
        except lang_detect_exception.LangDetectException:
            # Handle cases where langdetect fails to detect the language
            return self._handle_detection_failure(text)

    def _handle_detection_failure(self, text):
        # Implement your fallback strategy here
        # ...

        # For illustration, let's assume English as the default
        print(f"Failed to detect language for text: {text}")  # Log the failure for debugging
        return 'en'  # Return 'en' as the default language