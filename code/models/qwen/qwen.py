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
        self.session = Memory(initial_template="You are an AI assistant tasked with helping the user on film or series-related questions. Read the following data and conversation history and answer the question. If you cannot infer information from the data, say \"I don't know\".\n\nData: {data}\n\n{history}\nAssistant:")
        self._download_model_if_missing()

        #TODO prompt engineer this? it seems to return empty string way too often
        with open("./models/qwen/prompt_template_qwen.txt", "r") as fd:
            self.prompt_template = fd.read()

    # extract titles, people, ...?
    # return a dict of lists, data will be scraped for each element in each list
    def extract_keyphrases(self, prompt):
        out = {"movies": [], "people": [], "key": []}

        tagger = POStagger()
        tagged = tagger.tag(prompt)

        for key in tagged:
            # TODO add people, etc., remove dates (it tages 28 years etc.)
            # also, this doesn't work very well - look for alternatives?
            out["movies"].append(key)

        stop_words = set(nltk.corpus.stopwords.words("english"))
        prompt_tokens = nltk.tokenize.word_tokenize(prompt)
        output_text = [word for word in prompt_tokens if word not in stop_words]
        out["key"] = list(set(output_text))

        return out

    def train(self):
        pass

    def reply(self, prompt):
        return self.prompt_nonstream(prompt)

    def prompt_stream(self, prompt: str, data: str = "") -> Iterator[ollama.GenerateResponse]:
        rag = Rag(prompt, self.mode, self.datafolder, self.outname, self.sources)
        self.context, state = rag.get_context()

        #final_prompt = self.prompt_template.format(data=data, query=prompt)
        final_prompt = self.session.get_template(data, prompt)
        reply = ollama.generate(model=self.model_label, prompt=final_prompt, stream=True)

        thinking = True
        fullresponse = ""
        for i, response in enumerate(reply):
            if not thinking:
                fullresponse += response.response
            if response.response == "</think>":
                thinking = False

        # here we have an option not to remember a potentially bad answer (if we come up with a suitable metric)
        self.session.add(prompt, str(fullresponse))
    
        return fullresponse, state

    def prompt_nonstream(self, prompt: str, data: str = "") -> ollama.GenerateResponse:
        rag = Rag(prompt, self.mode, self.datafolder, self.outname, self.sources)
        self.context, state = rag.get_context()

        final_prompt = self.prompt_template.format(data=data, query=prompt)
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
