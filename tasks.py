from langgraph import Task, tool_code
import uuid

class Tasks:
    def __init__(self, config):
        self.config = config

    def fetch_new_emails_task(self):
        return Task(
            name="fetch_new_emails",
            tool_code=tool_code(
                """
                # 1. Fetch new emails from Outlook 365 (replace with your actual implementation)
                new_emails = fetch_new_emails_from_outlook()

                # 2. If there are no new emails, return None
                if not new_emails:
                    return None

                # 3. If there are new emails, return the first one (you can adjust this logic if needed)
                return new_emails[0]
                """
            ),
        )

    def check_registry_task(self):
        return Task(
            name="check_registry_and_extract_question",
            tool_code=tool_code(
                """
                # 1. Check if sender is in the registry
                if email.get('from') in email_registry:
                    # 2a. If registered, extract the question and proceed
                    extracted_question = extract_question_from_email(email)

                    # Generate a UID for this submission
                    submission_uid = str(uuid.uuid4())

                    # Check if it's a follow-up question
                    in_reply_to_uid = extract_uid_from_reply(email)
                    if in_reply_to_uid:
                        context = cache.get(in_reply_to_uid)
                        if context:
                            # Return follow-up question with context
                            return {"question": extracted_question, "uid": submission_uid, "context": context}
                        else:
                            # Handle case where context is not found
                            error_message = "Context not found. Please rephrase your question or start a new conversation."
                            send_error_email(email.get('from'), error_message, submission_uid)
                            return None

                    # Store new question in the database
                    store_interaction_in_db(submission_uid, email.get('from'), question=extracted_question)
                    return {"question": extracted_question, "uid": submission_uid, "context": None} 
                else:
                    # 2b. If not registered, send automated reply
                    reply_uid = str(uuid.uuid4())
                    send_registration_instructions(email.get('from'), reply_uid)
                    # Store automated reply interaction in the database
                    store_interaction_in_db(reply_uid, email.get('from'), status="registration_reply")
                    return None
                """
            ),
            args={"email": "{new_email}", "email_registry": self.config.email_registry, "cache": self.config.cache},
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
            args={"question": "{question}", "cache": self.config.cache, "translator": Translator(self.config.cache, self.config.tokenizers, self.config.models)},
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
                # ... (Implement answer generation logic, consider context if provided)
                answer = generate_answer_from_llm(query, docs, context)

                # Format references
                references = [f"Document: {doc.metadata['title']}" for doc in docs]
                answer_with_references = f"{answer}\n\nReferences:\n{chr(10).join(references)}"

                return answer_with_references
                """
            ),
            args={"query": "{question}", "docs": "{documents}", "context": "{context}"},
        )

    def self_correct_task(self):
        return Task(
            name="self_correct",
            tool_code=tool_code(
                """
                # Evaluate the answer 
                # ... (Implement your evaluation logic)

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
            args={"answer": "{answer}", "input_language": "{input_language}", "cache": self.config.cache, "translator": Translator(self.config.cache, self.config.tokenizers, self.config.models)},
        )

    def construct_email_response_task(self):
        return Task(
            name="construct_email_response",
            tool_code=tool_code(
                """
                # Include UID and problems (if any) in the email
                email_body = f"Answer:\n{answer}\n\nReference ID: {uid}"
                if problems:
                    email_body += f"\n\nIdentified problems: {problems}"
                return email_body
                """
            ), 
            args={"answer": "{answer}", "uid": "{uid}", "problems": "{problems}"},
        )

    def send_email_and_update_db_task(self):
        return Task(
            name="send_email_and_update_db",
            tool_code=tool_code(
                """
                # Send the email using the local mail server
                # ... (Implement your email sending logic)
                send_email(user_email, "Answer to your question", email_body)

                # Update the database
                status = "answered" if problems is None else "answered_with_problems"
                update_interaction_in_db(uid, answer, status)

                return "Email sent successfully."
                """
            ),
            args={"email_body": "{email_body}", "user_email": "{user_email}", "uid": "{uid}", "problems": "{problems}"},
        )