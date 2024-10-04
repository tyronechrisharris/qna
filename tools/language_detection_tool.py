import logging

from langchain.tools import BaseTool
import fasttext

class LanguageDetectionTool(BaseTool):
    """
    Tool for detecting the language of a given text using the 'fasttext' library.
    Includes error handling and logging.
    """

    name = "language_detection"
    description = "Detects the language of the given text."

    def __init__(self):
        """
        Initialize the LanguageDetectionTool with the FastText language model.
        """
        super().__init__()
        try:
            # Load the FastText language identification model
            self.model = fasttext.load_model('lid.176.bin')  # Replace with the actual path to your downloaded model
        except Exception as e:
            logging.error(f"Error loading FastText model: {e}")
            raise

    def _run(self, text: str) -> str:
        """
        Detect the language of the input text using FastText.

        Args:
            text: The text to detect the language of.

        Returns:
            The detected language code (e.g., 'en', 'es', 'fr').
        """
        try:
            # Use FastText to detect the language
            predictions = self.model.predict(text, k=1)  # Get the top prediction
            # Extract the language code from the prediction
            detected_language = predictions[0][0].replace("__label__", "")  
            return detected_language
        except Exception as e:
            logging.error(f"Error during language detection with FastText: {e}")
            return "en"  # Default to English if an error occurs