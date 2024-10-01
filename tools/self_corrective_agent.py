from langchain.tools import BaseTool
from sentence_transformers import SentenceTransformer, util
import spacy

class SelfCorrectiveAgent(BaseTool):
    """
    This tool checks the generated answer for hallucinations, coherence, and sense.
    """

    name = "self_corrective_agent"
    description = "Checks the generated answer for hallucinations, coherence, and sense."

    def __init__(self):
        """
        Initialize the SelfCorrectiveAgent with necessary models.
        """
        super().__init__()
        self.similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') 
        self.nlp = spacy.load("en_core_web_sm")

    def _run(self, query: str, answer: str, documents: list, retry_count: int = 0) -> tuple:
        """
        Evaluates the generated answer and provides feedback or passes it along with problems.

        Args:
            query: The original question asked by the user
            answer: The generated answer to be evaluated
            documents: The list of documents used to generate the answer
            retry_count: The number of times the system has already tried to generate a better answer

        Returns:
            A tuple containing:
                - The answer if it's valid, or None if it needs improvement
                - Feedback to the Document Retriever if retry_count < 3 and the answer is not valid
                - A description of the problems if retry_count >= 3 and the answer is not valid
        """

        if self.is_answer_valid(query, answer, documents):
            return answer, None  # Answer is acceptable
        else:
            if retry_count < 3:
                feedback = self.generate_feedback(query, answer, documents)
                return None, feedback  # Answer needs improvement
            else:
                problems_description = self.describe_problems(answer)
                return answer, problems_description  # Max retries reached

    def is_answer_valid(self, query: str, answer: str, documents: list) -> bool:
        """
        Checks if the answer is valid based on various criteria.

        Args:
            query: The original question
            answer: The generated answer
            documents: The list of documents used

        Returns:
            True if the answer is valid, False otherwise
        """

        # 1. Hallucination Check
        if not self.is_answer_grounded(query, answer, documents):
            return False 

        # 2. Coherence and Sense Check
        if not self.is_answer_coherent(answer):
            return False

        # 3. Factual Accuracy Check (Optional, if you have a fact-checking library)
        # if not check_facts(answer):
        #     return False

        return True

    def is_answer_grounded(self, query: str, answer: str, documents: list) -> bool:
        """
        Checks if the answer is grounded in the provided documents using semantic similarity.

        Args:
            query: The original question
            answer: The generated answer
            documents: The list of documents used

        Returns:
            True if the answer is grounded, False otherwise
        """

        query_embedding = self.similarity_model.encode(query)
        answer_embedding = self.similarity_model.encode(answer)
        doc_embeddings = self.similarity_model.encode([doc.page_content for doc in documents])

        # Calculate cosine similarities
        query_answer_sim = util.cos_sim(query_embedding, answer_embedding)
        answer_docs_sim = util.cos_sim(answer_embedding, doc_embeddings)

        # Check if answer is semantically similar to the query and at least one document
        return query_answer_sim >= 0.5 and answer_docs_sim.max() >= 0.3

    def is_answer_coherent(self, answer: str) -> bool:
        """
        Checks if the answer is coherent and makes sense using basic NLP techniques.

        Args:
            answer: The generated answer

        Returns:
            True if the answer is coherent, False otherwise
        """

        doc = self.nlp(answer)
        return doc._.coherence_score >= 0.5 and doc._.sentence_cohesion_score >= 0.5

    def generate_feedback(self, query: str, answer: str, documents: list) -> str:
        """
        Generates feedback to the Document Retriever based on the identified issues.

        Args:
            query: The original question
            answer: The generated answer
            documents: The list of documents used

        Returns:
            A feedback message explaining the problems with the answer
        """

        feedback = ""

        if not self.is_answer_grounded(answer, documents):
            feedback += "The answer seems to contain information not present in the provided documents. "

        if not self.is_answer_coherent(answer):
            feedback += "The answer is not clear or logically consistent. "

        # Optional: Add factual accuracy feedback if you have a fact-checking library

        feedback += "Please try to generate a more relevant and accurate answer based on the given documents."
        return feedback

    def describe_problems(self, answer: str) -> str:
        """
        Provides a concise description of the problems identified in the answer.

        Args:
            answer: The generated answer

        Returns:
            A comma-separated string listing the identified problems
        """

        problems = []

        if not self.is_answer_grounded(answer, documents):
            problems.append("Potential hallucinations (information not found in the documents)")

        if not self.is_answer_coherent(answer):
            problems.append("Incoherence or lack of clarity")

        # Optional: Add factual accuracy problems if you have a fact-checking library

        return ", ".join(problems)