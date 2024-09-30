from langchain.tools import BaseTool
from sentence_transformers import SentenceTransformer, util
import spacy

class SelfCorrectiveAgent(BaseTool):
    name = "self_corrective_agent"
    description = "Checks the generated answer for hallucinations, coherence, and sense."

    def __init__(self):
        super().__init__()
        self.similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') 
        self.nlp = spacy.load("en_core_web_sm")

    def _run(self, query, answer, documents, retry_count=0):
        if self.is_answer_valid(query, answer, documents):
            return answer, None  # Answer is acceptable
        else:
            if retry_count < 3:
                feedback = self.generate_feedback(query, answer, documents)
                return None, feedback  # Answer needs improvement
            else:
                problems_description = self.describe_problems(answer)
                return answer, problems_description  # Max retries reached

    def is_answer_valid(self, query, answer, documents):
        # 1. Hallucination Check
        if not self.is_answer_grounded(answer, documents):
            return False 

        # 2. Coherence and Sense Check
        if not self.is_answer_coherent(answer):
            return False

        # 3. Factual Accuracy Check (Optional, if you have a fact-checking library)
        # if not check_facts(answer):
        #     return False

        return True

    def is_answer_grounded(self, answer, documents):
        query_embedding = self.similarity_model.encode(query)
        answer_embedding = self.similarity_model.encode(answer)
        doc_embeddings = self.similarity_model.encode([doc.page_content for doc in documents])

        # Calculate cosine similarities
        query_answer_sim = util.cos_sim(query_embedding, answer_embedding)
        answer_docs_sim = util.cos_sim(answer_embedding, doc_embeddings)

        # Check if answer is semantically similar to the query and at least one document
        if query_answer_sim < 0.5 or answer_docs_sim.max() < 0.3:
            return False
        else:
            return True

    def is_answer_coherent(self, answer):
        doc = self.nlp(answer)
        if doc._.coherence_score < 0.5 or doc._.sentence_cohesion_score < 0.5:
            return False
        else:
            return True

    def generate_feedback(self, query, answer, documents):
        feedback = ""

        # 1. Hallucination Feedback
        if not self.is_answer_grounded(answer, documents):
            feedback += "The answer seems to contain information not present in the provided documents. "

        # 2. Coherence & Sense Feedback
        if not self.is_answer_coherent(answer):
            feedback += "The answer is not clear or logically consistent. "

        # 3. Factual Accuracy Feedback (Optional)
        # if not check_facts(answer):
        #     feedback += "Some facts in the answer might be inaccurate. "

        feedback += "Please try to generate a more relevant and accurate answer based on the given documents."
        return feedback

    def describe_problems(self, answer):
        problems = []

        # 1. Hallucination
        if not self.is_answer_grounded(answer, documents):
            problems.append("Potential hallucinations (information not found in the documents)")

        # 2. Coherence & Sense
        if not self.is_answer_coherent(answer):
            problems.append("Incoherence or lack of clarity")

        # 3. Factual Accuracy (Optional)
        # if not check_facts(answer):
        #     problems.append("Potential factual inaccuracies")

        return ", ".join(problems)