import asyncio
import logging
import queue
import time

from langgraph import Graph

from config import Config
from tasks import Tasks
from tools.language_detection_tool import LanguageDetectionTool
from tools.translator import Translator
from tools.document_answerer import DocumentAnswerer
from tools.self_corrective_agent import SelfCorrectiveAgent
from tools.chat_input_tool import (
    ChatInputTool,
    get_user_input_from_terminal,
    display_answer_in_terminal,
)

# Set up logging
logging.basicConfig(
    filename="qna_system.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
)


# 3. Main Program
class QuestionAnsweringSystem:
    def __init__(self, config):
        self.config = config
        self.request_queue = queue.Queue()
        self.tasks = Tasks(config)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = Graph()

        # Add tasks to the graph
        graph.add_tasks(
            [
                self.tasks.chat_input_task(),
                self.tasks.detect_language_task(),
                self.tasks.translate_to_english_task(),
                self.tasks.retrieve_documents_task(),
                self.tasks.generate_answer_task(),
                self.tasks.self_correct_task(),
                self.tasks.translate_to_user_language_task(),
                self.tasks.display_answer_in_chat_task(),
            ]
        )

        # Define the flow of the graph
        graph.connect(
            self.tasks.chat_input_task(),
            self.tasks.detect_language_task(),
            edge="question, uid, context",
        )
        graph.connect(
            self.tasks.detect_language_task(),
            self.tasks.translate_to_english_task(),
            edge="question, uid, context, input_language",
        )
        graph.connect(
            self.tasks.detect_language_task(),
            self.tasks.retrieve_documents_task(),
            edge="question, uid, context, input_language",
        )  # Connect directly if input_language is 'en'
        graph.connect(
            self.tasks.translate_to_english_task(),
            self.tasks.retrieve_documents_task(),
            edge="question, uid, context, input_language",
        )
        graph.connect(
            self.tasks.retrieve_documents_task(),
            self.tasks.generate_answer_task(),
            edge="question, uid, context, input_language, documents",
        )
        graph.connect(
            self.tasks.generate_answer_task(),
            self.tasks.self_correct_task(),
            edge="question, uid, context, input_language, documents, answer, retry_count=0",
        )

        # Self-correction loop
        graph.connect(
            self.tasks.self_correct_task(),
            self.tasks.translate_to_user_language_task(),
            edge="question, uid, context, input_language, documents, answer, problems",
        )  # When retry_count >= 3
        graph.connect(
            self.tasks.self_correct_task(),
            self.tasks.retrieve_documents_task(),
            edge="question, uid, context, input_language, feedback",
        )  # When retry_count < 3 and answer is not acceptable

        graph.connect(
            self.tasks.translate_to_user_language_task(),
            self.tasks.display_answer_in_chat_task(),
            edge="uid, input_language, answer, problems",
        )

        return graph

    async def run(self):
        while True:
            try:
                # 1. Check for new chat requests
                self.graph.execute(
                    inputs={
                        "request_queue": self.request_queue,
                        "cache": self.config.cache,
                        "current_chat_uid": None,
                    }
                )

                # 2. Process requests from the queue
                while not self.request_queue.empty():
                    request = self.request_queue.get()
                    _, question, uid, context, _ = request

                    # 3. Process the chat request
                    try:
                        input_language = language_detection_tool.run(question)
                    except Exception as e:
                        logging.error(f"Error during language detection: {e}")
                        display_answer_in_terminal(
                            "Error: Could not detect language."
                        )
                        continue

                    retry_count = 0

                    while retry_count < 3:
                        try:
                            if input_language != "en":
                                question = translator.run(question)
                        except Exception as e:
                            logging.error(f"Error during translation to English: {e}")
                            display_answer_in_terminal(
                                "Error: Could not translate to English."
                            )
                            break

                        try:
                            documents = document_retriever.run(question)
                            answer = document_answerer.run(
                                question, documents, context
                            )
                            (
                                answer,
                                feedback_or_problems,
                            ) = self_corrective_agent.run(
                                question, answer, documents, retry_count
                            )
                        except Exception as e:
                            logging.error(f"Error during answer generation: {e}")
                            display_answer_in_terminal(
                                "Error: Could not generate an answer."
                            )
                            break

                        if feedback_or_problems is None:
                            break

                        if isinstance(feedback_or_problems, str):
                            answer += (
                                f"\n\nIdentified problems: {feedback_or_problems}"
                            )
                            break

                        retry_count += 1
                        documents = document_retriever.run(
                            question, feedback_or_problems
                        )
                        answer = document_answerer.run(question, documents, context)

                    try:
                        if input_language != "en":
                            answer = translator.run(
                                answer, target_language=input_language
                            )
                    except Exception as e:
                        logging.error(
                            f"Error during translation to original language: {e}"
                        )
                        display_answer_in_terminal(
                            "Error: Could not translate to original language."
                        )
                        continue

                    # Display the answer in the chat interface
                    display_answer_in_terminal(answer)

                    # Update cache with context
                    self.config.cache.set(
                        uid, {"question": question, "documents": documents, "answer": answer}
                    )

                    # Update DB for chat interactions
                    status = (
                        "answered"
                        if feedback_or_problems is None
                        else "answered_with_problems"
                    )
                    self.tasks.update_interaction_in_db(uid, answer, status)

            except Exception as e:
                logging.error(f"An unexpected error occurred: {e}")

            time.sleep(1)  # Adjust the interval as needed

if __name__ == "__main__":
    config = Config()

    # Initialize tools
    language_detection_tool = LanguageDetectionTool()
    translator = Translator(config.cache, config.tokenizer, config.models)
    document_answerer = DocumentAnswerer(config.vectorstore, config.cache, config.llm)
    self_corrective_agent = SelfCorrectiveAgent()

    system = QuestionAnsweringSystem(config)
    asyncio.run(system.run())