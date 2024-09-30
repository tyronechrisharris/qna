import requests
from langchain.tools import BaseTool

class EmailOutputTool(BaseTool):
    name = "email_output"
    description = "Sends the final answer to the user via email."

    def __init__(self, smtp_server, smtp_port, sender_email, sender_password, db_cursor):
        super().__init__()
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.db_cursor = db_cursor

    def _run(self, answer, user_email, submission_uid, problems_description=None):
        # 1. Construct the email body
        email_body = self._construct_email_body(answer, submission_uid, problems_description)

        # 2. Send the email
        self._send_email(user_email, "Answer to your question", email_body)

        # 3. Update the database
        status = "answered" if problems_description is None else "answered_with_problems"
        self._update_interaction_in_db(submission_uid, answer, status)

        return "Email sent successfully."

    def _construct_email_body(self, answer, submission_uid, problems_description):
        body = f"Answer:\n{answer}\n\nReference ID: {submission_uid}"
        if problems_description:
            body += f"\n\nIdentified problems: {problems_description}"
        return body

    def _send_email(self, recipient_email, subject, body):
        # Placeholder: Implement your email sending logic using Outlook 365 API here
        # You'll likely need to use the 'requests' library to make API calls
        # Handle authentication, error handling, etc.
        # ...
        raise NotImplementedError("Implement _send_email using Outlook 365 API")

    def _update_interaction_in_db(self, uid, answer, status):
        self.db_cursor.execute('''
            UPDATE email_interactions
            SET answer = ?, status = ?
            WHERE uid = ?
        ''', (answer, status, uid))
        conn.commit()

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("EmailOutputTool does not support async")