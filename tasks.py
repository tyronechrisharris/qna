import uuid
import curses  # For the chat interface

from langgraph import Task, tool_code

class Tasks:
    def __init__(self, config):
        self.config = config

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
                translation = translator.run(question) 

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
                answer = generate_answer_from_llm(query, docs, context)

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
                update_interaction_in_db(uid, answer, status)
                """
            ),
            args={"answer": "{answer}", "problems": "{problems}", "uid": "{uid}", "cursor": self.config.cursor, "conn": self.config.conn},
        )

# Helper function for chat interface (using curses)
def get_user_input_from_terminal():
    # ... (Implementation using curses to get user input)
    pass

def display_answer_in_terminal(answer):
    # ... (Implementation using curses to display the answer)
    pass

def store_interaction_in_db(uid, question=None, answer=None, status="received", other_metadata=None):
    """
    This function should implement the logic to store the interaction details in your database.
    You'll likely need to use your database cursor to execute an INSERT query.

    Args:
        uid: The unique identifier for the interaction.
        question: The question submitted by the user (optional).
        answer: The answer generated by the system (optional).
        status: The status of the interaction (default is "received").
        other_metadata: Any additional metadata you want to store (optional).
    """

    cursor.execute('''
        INSERT INTO email_interactions (uid, timestamp, question, answer, status, other_metadata)
        VALUES (?, datetime('now'), ?, ?, ?, ?)
    ''', (uid, question, answer, status, other_metadata))
    conn.commit()

def update_interaction_in_db(uid, answer, status):
    """
    This function should implement the logic to update an existing interaction in your database.
    You'll likely need to use your database cursor to execute an UPDATE query.

    Args:
        uid: The unique identifier for the interaction.
        answer: The updated answer to be stored.
        status: The updated status of the interaction.
    """

    cursor.execute('''
        UPDATE email_interactions
        SET answer = ?, status = ?
        WHERE uid = ?
    ''', (answer, status, uid))
    conn.commit()