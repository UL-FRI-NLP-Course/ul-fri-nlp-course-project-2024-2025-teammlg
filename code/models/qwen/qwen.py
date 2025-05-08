from typing import Iterator
from ..model import Model
import ollama
from scraper import *
from POStagger import *
from summarizer import *
from rag import *
from memory import *

class QwenChatBot(Model):
    def __init__(self, name, folder, datafolder, sources=["tmdb", "letterboxd", "justwatch"], mode="naive"):
        self.name = name
        self.folder = folder
        self.datafolder = datafolder
        self.sources = sources
        self.model_label = "qwen:1.8b"
        self.outname = "qwen1_8"
        self.chat_history = []
        self.mode = mode
        self.context = None
        self.session = Memory(initial_template="You are an AI assistant tasked with helping the user on film or series-related questions. Read the following data and conversation history and answer the question. \n\nData: {data}\n\n{history}\nAssistant:")
        self._download_model_if_missing()

        with open("./models/qwen/prompt_template_qwen.txt", "r") as fd:
            self.prompt_template = fd.read()

    def train(self):
        pass

    def reply(self, prompt):
        return self.prompt_nonstream(prompt)

    def prompt_stream(self, prompt: str, data: str = "") -> Iterator[ollama.GenerateResponse]:
        rag = Rag(prompt, self.mode, self.datafolder, self.outname, self.sources)
        self.context, state = rag.get_context()

        final_prompt = self.session.get_template(self.context, prompt)
        reply = ollama.generate(model=self.model_label, prompt=final_prompt, stream=True)

        fullresponse = ""
        for i, response in enumerate(reply):
            fullresponse += response.response

        # here we have an option not to remember a potentially bad answer (if we come up with a suitable metric)
        self.session.add(prompt, str(fullresponse))
    
        return fullresponse, state

    def prompt_nonstream(self, prompt: str, data: str = "") -> ollama.GenerateResponse:
        rag = Rag(prompt, self.mode, self.datafolder, self.outname, self.sources)
        self.context, state = rag.get_context()

        final_prompt = self.prompt_template.format(data=self.context, query=prompt)
        return (
            ollama.generate(model=self.model_label, prompt=final_prompt, stream=False),
            state,
        )

    def _download_model_if_missing(self):
        """Checks if the model is already downloaded, and downloads it otherwise"""
        all_local_models = ollama.list()
        for model in all_local_models.models:
            if model.model == self.model_label:
                return  # We found the model - we exit
        print(f"Could not find local '{self.model_label}' instance, downloading...")
        response = ollama.pull(self.model_label)
        print(response.completed)
