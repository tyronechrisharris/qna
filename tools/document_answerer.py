from langchain.tools import BaseTool
from langchain.chains import RetrievalQA
# Replace with your offline LLM interface
from langchain.llms import OpenAI 

class DocumentAnswerer(BaseTool):
    name = "document_answerer"
    description = "Answers questions based on the documents provided"

    def __init__(self, vectorstore, cache, llm):
        super().__init__()
        self.vectorstore = vectorstore
        self.cache = cache
        self.llm = llm  # Your offline LLM instance

    def _run(self, query, context=None):
        # 1. Check cache for existing answer
        cache_key = self._get_cache_key(query, context)
        cached_answer = self.cache.get(cache_key)
        if cached_answer:
            return cached_answer

        # 2. Retrieve relevant documents
        retriever = self.vectorstore.as_retriever()
        if context:
            # If context is provided (follow-up question), include it in the retrieval
            docs = retriever.get_relevant_documents(query, context)
        else:
            docs = retriever.get_relevant_documents(query)

        # 3. Generate answer using the LLM
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm, 
            chain_type="stuff",  # Adjust chain_type if needed for your LLM
            retriever=retriever, 
            return_source_documents=True  # To get the documents used for references
        )
        result = qa_chain({"query": query, "chat_history": context})  # Include context if available

        # 4. Format references
        used_documents = result["source_documents"]
        references = [f"Document: {doc.metadata['title']}" for doc in used_documents]
        answer_with_references = f"{result['result']}\n\nReferences:\n{chr(10).join(references)}"

        # 5. Store answer in cache
        self.cache.set(cache_key, answer_with_references)

        return answer_with_references

    def _get_cache_key(self, query, context=None):
        # Generate a unique cache key based on the query and context
        if context:
            return f"{query}_{hash(str(context))}" 
        else:
            return query