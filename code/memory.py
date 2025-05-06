class Memory():
    def __init__(self, initial_template=None, buffer_len=10):
        self.buffer_len = buffer_len
        self.queries = []
        self.replies = []

        if initial_template:
            self.template = initial_template
        else:
            self.template = "You are an AI assistant tasked with helping the user on film or series-related questions. Read the following data and conversation history and answer the question. If you cannot infer information from the data, do not answer the question.\n\nData: {data}\n\n{history}\nAssistant:"

    # add a new query and reply into the memory
    def add(self, query, reply):
        if len(self.queries) == self.buffer_len:
            self.queries = self.queries[1:] + [query]
            self.replies = self.replies[1:] + [reply]
        else:
            self.queries.append(query)
            self.replies.append(reply)

    def get_history(self):
        if len(self.queries) == 0:
            return ""
        history = ""
        for i, (query, reply) in enumerate(zip(self.queries, self.replies)):
            # numbering the queries and replies probably doesn't accomplish anything, but we have i if we wanna experiment
            history += "User: " + query.strip() + "\n"
            history += "Assistant: " + reply.strip() + "\n"
        return history

    def get_template(self, context, new_query):
        history = self.get_history() + "User: " + new_query.strip() + "\n"
        return self.template.format(data=context, history=history)
