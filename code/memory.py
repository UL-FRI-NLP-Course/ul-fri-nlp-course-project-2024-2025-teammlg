from typing import Dict, List


class Memory():
    def __init__(self, initial_template=None, buffer_len=10):
        self.buffer_len = buffer_len
        # self.queries = []
        # self.replies = []
        self.permanent_messages = []
        self.chat = []

        if initial_template:
            self.template = initial_template
        else:
            self.template = "You are an AI assistant tasked with helping the user on film or series-related questions. Read the following data and conversation history and answer the question. If you cannot infer information from the data, do not answer the question.\n\nData: {data}\n\n{history}\nAssistant:"

    def add_user_query(self, query: str):
        if len(self.chat) >= self.buffer_len:
            n = self.buffer_len - 1
            self.chat = self.chat[-n:]
        self.chat.append({
            "role": "user",
            "content": query
        })
    
    def add_assistant_response(self, response: str):
        if len(self.chat) >= self.buffer_len:
            n = self.buffer_len - 1
            self.chat = self.chat[-n:]
        self.chat.append({
            "role": "assistant",
            "content": response
        })
    
    def add_system_message(self, message: str, permament: bool = True):
        if len(self.chat) >= self.buffer_len:
            n = self.buffer_len - 1
            self.chat = self.chat[-n:]
        if permament:
            self.permanent_messages.append({
                "role": "system",
                "content": message
            })
        else:
            self.chat.append({
                "role": "system",
                "content": message
            })

    def get_chat_history(self) -> List[Dict]:
        #history = self.get_history() + "User: " + new_query.strip() + "\n"
        chat = self.permanent_messages + self.chat
        return chat
