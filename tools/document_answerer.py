import logging
from typing import Optional, List

from langchain.tools import BaseTool
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.llms import BaseLLM  # Import the base LLM class

class DocumentAnswerer(BaseTool):
    """
    This tool answers questions based on the documents provided.
    """

    name = "document_answerer"
    description = "Answers questions based on the documents provided"

    def __init__(self, vectorstore, cache, llm: BaseLLM):
        """
        Initialize the DocumentAnswerer with necessary components.

        Args:
            vectorstore: The vectorstore containing the embedded documents.
            cache: The cache object for storing and retrieving answers.
            llm: The offline LLM instance for generating answers
        """
        super().__init__()
        self.vectorstore = vectorstore
        self.cache = cache
        self.llm = llm

    def _run(self, query: str, context: Optional[List[dict]] = None) -> str:
        """
        Answer the query based on the provided documents and context.

        Args:
            query: The question to be answered
            context: Optional context from previous interactions (for follow-up questions)

        Returns:
            The answer to the query, including references to the documents used
        """
        try:
            # 1. Check cache for existing answer
            cache_key = self._get_cache_key(query, context)
            cached_answer = self.cache.get(cache_key)
            if cached_answer:
                return cached_answer.decode('utf-8')  # Decode from bytes to string

            # 2. Retrieve relevant documents
            retriever = self.vectorstore.as_retriever()
            if context:
                # If context is provided, include it in the retrieval
                docs = retriever.get_relevant_documents(query, context)
            else:
                docs = retriever.get_relevant_documents(query)

            # 3. Generate answer using the LLM
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm, 
                chain_type="stuff",  # Adjust chain_type if needed for your LLM
                retriever=retriever, 
                return_source_documents=True 
            )
            result = qa_chain({"query": query, "chat_history": context}) 

            # 4. Format references
            used_documents = result["source_documents"]
            references = [f"Document: {doc.metadata['title']}" for doc in used_documents]
            answer_with_references = f"{result['result']}\n\nReferences:\n{chr(10).join(references)}"

            # 5. Store answer in cache (store as bytes)
            self.cache.set(cache_key, answer_with_references.encode('utf-8'))

            return answer_with_references

        except Exception as e:
            logging.error(f"Error in DocumentAnswerer: {e}")
            return "Error: Could not generate an answer."

    def _get_cache_key(self, query: str, context: Optional[List[dict]] = None) -> str:
        """
        Generate a unique cache key based on the query and context

        Args:
            query: The question to be answered
            context: Optional context from previous interactions

        Returns:
            A unique string representing the cache key
        """

        if context:
            return f"{query}_{hash(str(context))}" 
        else:
            return query