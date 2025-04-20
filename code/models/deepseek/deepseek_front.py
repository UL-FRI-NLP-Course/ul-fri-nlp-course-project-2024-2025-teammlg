from typing import Iterator
from ..model import Model
import ollama
from scraper import *

class DeepSeekFilmChatBot(Model):
    def __init__(self, name, folder, datafolder, sources=["tmdb", "letterboxd"]):
        self.name = name
        self.folder = folder
        self.datafolder = datafolder
        self.sources = sources
        self.model_label = "deepseek-r1:1.5b"  # The name of the model for Ollama to download (all models here: https://ollama.com/search)
        self.chat_history = []
        self._download_model_if_missing()

        with open("./models/deepseek/prompt_template_deepseek.txt", "r") as fd:
            self.prompt_template = fd.read()

    def extract_title(self, prompt):
        #TODO
        #also, multiple titles? other phrases of interest?
        return "challengers"

    def rag(self, title):
        s = Scraper(title, self.sources)
        return s.data

    def train(self):
        pass

    def reply(self, prompt):
        return self.prompt_nonstream(prompt)

    def prompt_stream(
        self, prompt: str, data: str = ""
    ) -> Iterator[ollama.GenerateResponse]:
        """Feeds the prompt to the model, returning its response as a stream iterator"""
        final_prompt = self.prompt_template.format(data=data, query=prompt)
        return ollama.generate(model=self.model_label, prompt=final_prompt, stream=True)

    def prompt_nonstream(self, prompt: str, data: str = "") -> ollama.GenerateResponse:
        title = self.extract_title(prompt)
        self.rag(title)

        #TODO we might not always have both
        #context1 = open("data/scraped_data/tmdb_out.json").read()
        #context2 = open("data/scraped_data/letterboxd_out.json").read()
        for source in self.sources:
            context = open("data/scraped_data/"+source+"_out.json").read()
            data += context

        """Feeds the prompt to the model, returning its response"""
        final_prompt = self.prompt_template.format(data=data, query=prompt)
        return ollama.generate(
            model=self.model_label, prompt=final_prompt, stream=False
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
