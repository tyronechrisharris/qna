import time

from langgraph import Graph

from config import Config
from tasks import Tasks

# 3. Main Program (Refactored into a class using Langgraph)

class QuestionAnsweringSystem:
    def __init__(self, config):
        self.config = config
        self.tasks = Tasks(config)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = Graph()

        # Add tasks to the graph using the methods from the Tasks class
        graph.add_tasks([
            self.tasks.fetch_new_emails_task(), 
            self.tasks.check_registry_task(), 
            self.tasks.detect_language_task(),
            self.tasks.translate_to_english_task(), 
            self.tasks.retrieve_documents_task(),
            self.tasks.generate_answer_task(), 
            self.tasks.self_correct_task(), 
            self.tasks.translate_to_user_language_task(), 
            self.tasks.construct_email_response_task(),
            self.tasks.send_email_and_update_db_task()
        ])

        # Define the flow of the graph
        graph.connect(self.tasks.fetch_new_emails_task(), self.tasks.check_registry_task(), edge="new_email")
        graph.connect(self.tasks.check_registry_task(), self.tasks.detect_language_task(), edge="question, uid, context")
        graph.connect(self.tasks.detect_language_task(), self.tasks.translate_to_english_task(), edge="question, uid, context, input_language")
        graph.connect(self.tasks.detect_language_task(), self.tasks.retrieve_documents_task(), edge="question, uid, context, input_language")  # Connect directly if input_language is 'en'
        graph.connect(self.tasks.translate_to_english_task(), self.tasks.retrieve_documents_task(), edge="question, uid, context, input_language")
        graph.connect(self.tasks.retrieve_documents_task(), self.tasks.generate_answer_task(), edge="question, uid, context, input_language, documents")
        graph.connect(self.tasks.generate_answer_task(), self.tasks.self_correct_task(), edge="question, uid, context, input_language, documents, answer, retry_count=0")

        # Self-correction loop
        graph.connect(self.tasks.self_correct_task(), self.tasks.translate_to_user_language_task(), edge="question, uid, context, input_language, documents, answer, problems")  # When retry_count >= 3
        graph.connect(self.tasks.self_correct_task(), self.tasks.retrieve_documents_task(), edge="question, uid, context, input_language, feedback")  # When retry_count < 3 and answer is not acceptable

        graph.connect(self.tasks.translate_to_user_language_task(), self.tasks.construct_email_response_task(), edge="uid, input_language, answer, problems")
        graph.connect(self.tasks.construct_email_response_task(), self.tasks.send_email_and_update_db_task(), edge="email_body, uid, user_email, problems")
        
        return graph

    def run(self):
        while True:
            # Execute the graph 
            self.graph.execute(inputs={"email_registry": self.config.email_registry, "cache": self.config.cache})
            time.sleep(60)  # Adjust the interval as needed

if __name__ == "__main__":
    config = Config()
    system = QuestionAnsweringSystem(config)
    system.run()