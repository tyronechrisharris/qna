import curses
import uuid

from langchain.tools import BaseTool

class ChatInputTool(BaseTool):
    name = "check_chat"
    description = "Gets user input from the terminal-based chat interface"

    def __init__(self, request_queue):
        super().__init__()
        self.request_queue = request_queue
        self.current_chat_uid = None

    def _run(self):
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
        self._display_message(chat_win, "Welcome to the Offline Q&A Chatbot!")

        while True:
            # Get user input
            user_input = self._get_user_input(input_win)

            if user_input:
                # Check if it's a new chat session
                if self.current_chat_uid is None:
                    self.current_chat_uid = str(uuid.uuid4())

                # Retrieve context if it's a follow-up question
                context = cache.get(self.current_chat_uid) if self.current_chat_uid else None

                # Add the request to the queue
                self.request_queue.put(("chat", user_input, self.current_chat_uid, context, None))

                # Display user input in the chat window
                self._display_message(chat_win, f"You: {user_input}")

    def _get_user_input(self, input_win):
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

    def _display_message(self, chat_win, message):
        chat_win.addstr(message + "\n")
        chat_win.refresh()

        # Scroll the chat window if necessary
        max_y, _ = chat_win.getmaxyx()
        if chat_win.getyx()[0] >= max_y - 1:
            chat_win.scroll(1)