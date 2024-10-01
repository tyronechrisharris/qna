import time
import queue

from langgraph import Graph

from config import Config
from tasks import Tasks
from tools.language_detection_tool import LanguageDetectionTool
from tools.translator import Translator
from tools.document_answerer import DocumentAnswerer
from tools.self_corrective_agent import SelfCorrectiveAgent
from tools.chat_input_tool import ChatInputTool, get_user_input_from_terminal, display_answer_in_terminal

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
        graph.add_tasks([
            self.tasks.chat_input_task(), 
            self.tasks.detect_language_task(),
            self.tasks.translate_to_english_task(), 
            self.tasks.retrieve_documents_task(),
            self.tasks.generate_answer_task(), 
            self.tasks.self_correct_task(), 
            self.tasks.translate_to_user_language_task(), 
            self.tasks.display_answer_in_chat_task()
        ])

        # Define the flow of the graph
        graph.connect(self.tasks.chat_input_task(), self.tasks.detect_language_task(), edge="question, uid, context")
        graph.connect(self.tasks.detect_language_task(), self.tasks.translate_to_english_task(), edge="question, uid, context, input_language")
        graph.connect(self.tasks.detect_language_task(), self.tasks.retrieve_documents_task(), edge="question, uid, context, input_language") 
        graph.connect(self.tasks.translate_to_english_task(), self.tasks.retrieve_documents_task(), edge="question, uid, context, input_language")
        graph.connect(self.tasks.retrieve_documents_task(), self.tasks.generate_answer_task(), edge="question, uid, context, input_language, documents")
        graph.connect(self.tasks.generate_answer_task(), self.tasks.self_correct_task(), edge="question, uid, context, input_language, documents, answer, retry_count=0")

        # Self-correction loop
        graph.connect(self.tasks.self_correct_task(), self.tasks.translate_to_user_language_task(), edge="question, uid, context, input_language, documents, answer, problems") 
        graph.connect(self.tasks.self_correct_task(), self.tasks.retrieve_documents_task(), edge="question, uid, context, input_language, feedback") 

        graph.connect(self.tasks.translate_to_user_language_task(), self.tasks.display_answer_in_chat_task(), edge="uid, input_language, answer, problems")
        
        return graph

    def run(self):
        while True:
            # 1. Check for new chat requests
            self.graph.execute(inputs={
                "request_queue": self.request_queue, 
                "cache": self.config.cache
            })

            # 2. Process requests from the queue
            while not self.request_queue.empty():
                request = self.request_queue.get()
                _, question, uid, context, _ = request  # Ignore 'source' and 'user_email' 

                # 3. Process the chat request
                input_language = language_detection_tool.run(question)
                retry_count = 0

                while retry_count < 3:
                    if input_language != 'en':
                        question = translator.run(question) 

                    documents = document_retriever.run(question)
                    answer = document_answerer.run(question, documents, context)
                    answer, feedback_or_problems = self_corrective_agent.run(question, answer, documents, retry_count)

                    if feedback_or_problems is None:
                        break 

                    if isinstance(feedback_or_problems, str):
                        answer += f"\n\nIdentified problems: {feedback_or_problems}"
                        break

                    retry_count += 1
                    documents = document_retriever.run(question, feedback_or_problems)
                    answer = document_answerer.run(question, documents, context)

                if input_language != 'en':
                    answer = translator.run(answer, target_language=input_language) 

                # Display the answer in the chat interface
                display_answer_in_terminal(answer)

                # Update cache with context for potential follow-up questions
                self.config.cache.set(uid, {"question": question, "documents": documents, "answer": answer})

                # Update DB for chat interactions
                status = "answered" if feedback_or_problems is None else "answered_with_problems"
                self.tasks.update_interaction_in_db(uid, answer, status)

            time.sleep(1)  # Adjust the interval as needed

if __name__ == "__main__":
    config = Config()

    # Initialize tools
    language_detection_tool = LanguageDetectionTool()
    translator = Translator(config.cache, config.tokenizer, config.models)
    document_answerer = DocumentAnswerer(config.vectorstore, config.cache, config.llm)
    self_corrective_agent = SelfCorrectiveAgent()

    system = QuestionAnsweringSystem(config)
    system.run()