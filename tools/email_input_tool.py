import uuid
import re
import sqlite3  # Or your preferred database library

from langchain.tools import BaseTool

class EmailInputTool(BaseTool):
    name = "email_input"
    description = "Retrieves user questions from email."

    def __init__(self, email_registry, db_cursor, cache):
        super().__init__()
        self.email_registry = email_registry
        self.db_cursor = db_cursor
        self.cache = cache

    def _run(self):
        # 1. Fetch new emails from the local mail server
        new_emails = fetch_new_emails_from_outlook()

        for email in new_emails:
            sender_email = email.get('from')

            # 2. Check if the sender is in the registry
            if sender_email in self.email_registry:
                # 3a. If registered, process the email
                extracted_question = self._extract_question_from_email(email)
                submission_uid = str(uuid.uuid4())

                in_reply_to_uid = self._extract_uid_from_reply(email)
                if in_reply_to_uid:
                    # Handle follow-up question
                    context = self.cache.get(in_reply_to_uid)
                    if context:
                        # Store follow-up question in the database
                        self._store_interaction_in_db(
                            submission_uid, sender_email, question=extracted_question, 
                            status="follow_up", other_metadata={"in_reply_to_uid": in_reply_to_uid}
                        )
                        return {"question": extracted_question, "uid": submission_uid, "context": context}
                    else:
                        # Handle case where context is not found
                        error_message = "Context not found. Please rephrase your question or start a new conversation."
                        self._send_error_email(sender_email, error_message, submission_uid)
                        return None  # No question to process

                # Store new question in the database
                self._store_interaction_in_db(submission_uid, sender_email, question=extracted_question)
                return {"question": extracted_question, "uid": submission_uid, "context": None}
            else:
                # 3b. If not registered, send an automated reply
                reply_uid = str(uuid.uuid4())
                self._send_registration_instructions(sender_email, reply_uid)
                # Store automated reply interaction in the database
                self._store_interaction_in_db(reply_uid, sender_email, status="registration_reply")

        # If no new emails from registered users, return None
        return None

    def _extract_question_from_email(self, email):
        # Placeholder: Implement your email parsing logic here to extract the question from the email body
        # You'll likely need to use the Microsoft Graph API or a suitable library to access the email content
        # ...
        raise NotImplementedError("Implement _extract_question_from_email using Outlook 365 API")

    def _extract_uid_from_reply(self, email):
        # Placeholder: Implement your logic to extract the UID from a reply email
        # You might look for a specific pattern or header in the email
        # ...
        raise NotImplementedError("Implement _extract_uid_from_reply using Outlook 365 API")

    def _send_registration_instructions(self, email_address, reply_uid):
        # Placeholder: Construct and send the automated registration email using Outlook 365 API
        # Include the reply_uid in the email body
        # ...
        raise NotImplementedError("Implement _send_registration_instructions using Outlook 365 API")

    def _send_error_email(self, email_address, error_message, submission_uid):
        # Placeholder: Construct and send an error email to the user using Outlook 365 API
        # Include the submission_uid in the email body
        # ...
        raise NotImplementedError("Implement _send_error_email using Outlook 365 API")

    def _store_interaction_in_db(self, uid, sender_email, question=None, answer=None, status="received", other_metadata=None):
        self.db_cursor.execute('''
            INSERT INTO email_interactions (uid, sender_email, timestamp, question, answer, status, other_metadata)
            VALUES (?, ?, datetime('now'), ?, ?, ?, ?)
        ''', (uid, sender_email, question, answer, status, other_metadata))
        conn.commit()