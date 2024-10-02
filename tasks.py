import uuid
import curses

from langgraph import Task, tool_code
from langdetect import detect
from tools.translator import Translator
from sentence_transformers import SentenceTransformer, util
import spacy
from langchain.chains import RetrievalQA

class Tasks:
    def __init__(self, config):
        self.config = config
        self.similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.nlp = spacy.load("en_core_web_sm")

    def chat_input_task(self):
        return Task(
            name="check_chat",
            tool_code=tool_code(
                """
                # 1. Get user input from the terminal (using curses)
                user_input = get_user_input_from_terminal()

                if user_input:
                    # 2. Check if this is a new chat session or a follow-up
                    if current_chat_uid is None:  # New session
                        # Generate a new UID for this chat session
                        current_chat_uid = str(uuid.uuid4())
                        # Display a message indicating a new session
                        display_message(chat_win, f"New chat session started (UID: {current_chat_uid})")

                    # 3. Retrieve context if it's a follow-up question
                    context = cache.get(current_chat_uid) if current_chat_uid else None

                    # 4. Add the request to the queue
                    request_queue.put(("chat", user_input, current_chat_uid, context, None))  

                return None  # Return None if no user input
                """
            ),
            args={"request_queue": "{request_queue}", "cache": self.config.cache},
        )

    def detect_language_task(self):
        return Task(
            name="detect_language",
            tool_code=tool_code(
                """
                detected_language = detect(question)
                return detected_language
                """
            ),
            args={"question": "{question}"},
        )

    def translate_to_english_task(self):
        return Task(
            name="translate_to_english",
            tool_code=tool_code(
                """
                # 1. Check cache for existing translation
                cached_translation = cache.get(question)
                if cached_translation:
                    return cached_translation

                # 2. Perform translation if not cached
                translation = translator.run(question)  # Use the translator tool

                # 3. Store translation in cache
                cache.set(question, translation)

                return translation
                """
            ),
            args={"question": "{question}", "cache": self.config.cache, "translator": Translator(self.config.cache, self.config.tokenizer, self.config.models)},
        )

    def retrieve_documents_task(self):
        return Task(
            name="retrieve_documents",
            tool_code=tool_code(
                """
                # Retrieve relevant documents
                docs = vectorstore.similarity_search(query)
                return docs
                """
            ),
            args={"query": "{question}", "vectorstore": self.config.vectorstore},
        )

    def generate_answer_task(self):
        return Task(
            name="generate_answer",
            tool_code=tool_code(
                """
                # Generate answer using the offline LLM
                answer = generate_answer_from_llm(query, docs, context, llm)

                # Format references
                references = [f"Document: {doc.metadata['title']}" for doc in docs]
                answer_with_references = f"{answer}\n\nReferences:\n{chr(10).join(references)}"

                return answer_with_references
                """
            ),
            args={"query": "{question}", "docs": "{documents}", "context": "{context}", "llm": self.config.llm},
        )

    def self_correct_task(self):
        return Task(
            name="self_correct",
            tool_code=tool_code(
                """
                # Evaluate the answer 
                if is_answer_valid(query, answer, documents):
                    return {"answer": answer, "feedback": None} 
                else:
                    if retry_count < 3:
                        feedback = generate_feedback(query, answer, documents)
                        return {"answer": None, "feedback": feedback}
                    else:
                        problems_description = describe_problems(answer)
                        return {"answer": answer, "problems": problems_description}
                """
            ),
            args={"query": "{question}", "answer": "{answer}", "documents": "{documents}", "retry_count": "{retry_count}"},
        )

    def translate_to_user_language_task(self):
        return Task(
            name="translate_to_user_language",
            tool_code=tool_code(
                """
                # 1. Check cache for existing translation
                cached_translation = cache.get(answer)
                if cached_translation:
                    return cached_translation

                # 2. Perform translation if not cached
                translation = translator.run(answer, target_language=input_language) 

                # 3. Store translation in cache
                cache.set(answer, translation)

                return translation
                """
            ),
            args={"answer": "{answer}", "input_language": "{input_language}", "cache": self.config.cache, "translator": Translator(self.config.cache, self.config.tokenizer, self.config.models)},
        )

    def display_answer_in_chat_task(self):
        return Task(
            name="display_answer_in_chat",
            tool_code=tool_code(
                """
                # Display the answer in the chat interface (using curses)
                display_answer_in_terminal(answer)

                # Update DB for chat interactions
                status = "answered" if problems is None else "answered_with_problems"
                update_interaction_in_db(uid, answer, status, cursor, conn)
                """
            ),
            args={"answer": "{answer}", "problems": "{problems}", "uid": "{uid}", "cursor": self.config.cursor, "conn": self.config.conn},
        )

# Helper functions for chat interface (using curses)
def get_user_input_from_terminal():
    # Initialize curses
    stdscr = curses.initscr()
    curses.cbreak()  # Get characters immediately
    curses.noecho()  # Don't echo user input
    stdscr.keypad(True)  # Enable special keys

    # Create chat window and input area
    height, width = stdscr.getmaxyx()
    chat_win = curses.newwin(height - 3, width, 0, 0)  # Leave space for input
    input_win = curses.newwin(3, width, height - 3, 0)
    chat_win.scrollok(True)  # Enable scrolling in chat window

    # Display welcome message
    display_message(chat_win, "Welcome to the Offline Q&A Chatbot!")

    # Get user input
    user_input = get_input(input_win)

    if user_input:
        # Display user input in the chat window
        display_message(chat_win, f"You: {user_input}")

    return user_input.strip(), chat_win

def get_input(input_win):
    input_win.clear()
    input_win.addstr(1, 0, "You: ")
    input_win.refresh()

    user_input = ""
    while True:
        key = input_win.getch()
        if key == curses.KEY_ENTER or key in [10, 13]:  # Enter key
            break
        elif key == curses.KEY_BACKSPACE or key == 127:  # Backspace
            if user_input:
                user_input = user_input[:-1]
                y, x = input_win.getyx()
                input_win.delch(y, x - 1)
                input_win.refresh()
        else:
            user_input += chr(key)
            input_win.addch(key)
            input_win.refresh()

    return user_input.strip()

def display_answer_in_terminal(answer):
    # Get the curses window from get_user_input_from_terminal
    stdscr = curses.initscr()
    height, width = stdscr.getmaxyx()
    chat_win = curses.newwin(height - 3, width, 0, 0)  # Leave space for input
    chat_win.scrollok(True)

    # Display the answer
    display_message(chat_win, f"Bot: {answer}")

def display_message(chat_win, message):
    chat_win.addstr(message + "\n")
    chat_win.refresh()

    # Scroll the chat window if necessary
    max_y, _ = chat_win.getmaxyx()
    if chat_win.getyx()[0] >= max_y - 1:
        chat_win.scroll(1)

def store_interaction_in_db(uid, question=None, answer=None, status="received", other_metadata=None, cursor=None, conn=None):
    """
    Stores the interaction details in the database.

    Args:
        uid: The unique identifier for the interaction.
        question: The question submitted by the user (optional).
        answer: The answer generated by the system (optional).
        status: The status of the interaction (default is "received").
        other_metadata: Any additional metadata you want to store (optional).
        cursor: The database cursor object.
        conn: The database connection object.
    """
    cursor.execute('''
        INSERT INTO chat_interactions (uid, timestamp, question, answer, status, other_metadata)
        VALUES (?, datetime('now'), ?, ?, ?, ?)
    ''', (uid, question, answer, status, other_metadata))
    conn.commit()

def update_interaction_in_db(uid, answer, status, cursor, conn):
    """
    Updates an existing interaction in the database.

    Args:
        uid: The unique identifier for the interaction.
        answer: The updated answer to be stored.
        status: The updated status of the interaction.
        cursor: The database cursor object.
        conn: The database connection object.
    """
    cursor.execute('''
        UPDATE chat_interactions
        SET answer = ?, status = ?
        WHERE uid = ?
    ''', (answer, status, uid))
    conn.commit()

def generate_answer_from_llm(query: str, docs: list, context: Optional[List[dict]], llm) -> str:
    """
    Generates an answer to the given query using the provided documents and context.

    Args:
        query: The question to be answered.
        docs: A list of relevant documents retrieved from the vectorstore.
        context: Optional context from previous interactions (for follow-up questions).
        llm: The offline LLM instance for generating answers.

    Returns:
        The generated answer to the query.
    """

    # Use RetrievalQA chain to generate answer
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff",  # Adjust chain_type if needed for your LLM
        retriever=self.config.vectorstore.as_retriever(), 
        return_source_documents=True 
    )
    result = qa_chain({"query": query, "chat_history": context}) 

    return result['result']

def is_answer_valid(query: str, answer: str, documents: list) -> bool:
    """
    Checks if the generated answer is valid based on various criteria.

    Args:
        query: The original question asked by the user.
        answer: The generated answer to be evaluated.
        documents: The list of documents used to generate the answer.

    Returns:
        True if the answer is valid, False otherwise.
    """

    # 1. Check if the answer is relevant to the query
    query_embedding = self.similarity_model.encode(query)
    answer_embedding = self.similarity_model.encode(answer)
    if util.cos_sim(query_embedding, answer_embedding)[0][0] < 0.6:
        return False

    # 2. Check if the answer is grounded in the provided documents
    doc_embeddings = self.similarity_model.encode([doc.page_content for doc in documents])
    if util.cos_sim(answer_embedding, doc_embeddings).max() < 0.3:
        return False

    # 3. Check if the answer is coherent and makes sense
    doc = self.nlp(answer)
    if doc._.coherence_score < 0.5 or doc._.sentence_cohesion_score < 0.5:
        return False

    return True

def generate_feedback(query: str, answer: str, documents: list) -> str:
    """
    Generates feedback to guide the Document Retriever in case the answer is not valid.

    Args:
        query: The original question asked by the user.
        answer: The generated answer to be evaluated.
        documents: The list of documents used to generate the answer.

    Returns:
        A string containing feedback for the Document Retriever.
    """

    feedback = "The answer needs improvement. Please consider the following:\n"

    # 1. Relevance feedback
    similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    query_embedding = similarity_model.encode(query)
    answer_embedding = similarity_model.encode(answer)
    if util.cos_sim(query_embedding, answer_embedding)[0][0] < 0.6:
        feedback += "- The answer doesn't seem to be relevant to the question.\n"

    # 2. Grounding feedback
    doc_embeddings = similarity_model.encode([doc.page_content for doc in documents])
    if util.cos_sim(answer_embedding, doc_embeddings).max() < 0.3:
        feedback += "- The answer might contain information not found in the provided documents.\n"

    # 3. Coherence and sense feedback
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(answer)
    if doc._.coherence_score < 0.5 or doc._.sentence_cohesion_score < 0.5:
        feedback += "- The answer is not clear or logically consistent.\n"

    return feedback

def describe_problems(answer: str) -> str:
    """
    Provides a description of the problems identified in the answer.

    Args:
        answer: The generated answer to be evaluated.

    Returns:
        A string describing the problems found in the answer.
    """
    problems = []

    # 1. Relevance problem
    similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    query_embedding = similarity_model.encode(query)
    answer_embedding = similarity_model.encode(answer)
    if util.cos_sim(query_embedding, answer_embedding)[0][0] < 0.6:
        problems.append("The answer doesn't seem to be relevant to the question.")

    # 2. Grounding problem
    doc_embeddings = similarity_model.encode([doc.page_content for doc in documents])
    if util.cos_sim(answer_embedding, doc_embeddings).max() < 0.3:
        problems.append("The answer might contain information not found in the provided documents.")

    # 3. Coherence and sense problem
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(answer)
    if doc._.coherence_score < 0.5 or doc._.sentence_cohesion_score < 0.5:
        problems.append("The answer is not clear or logically consistent.")

    return ", ".join(problems)