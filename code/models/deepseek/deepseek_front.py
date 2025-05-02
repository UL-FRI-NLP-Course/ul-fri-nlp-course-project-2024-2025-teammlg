from typing import Iterator, Tuple

import nltk
from ..model import Model
import ollama
from scraper import *
from POStagger import *
from summarizer import *


class DeepSeekFilmChatBot(Model):
    def __init__(
        self,
        name,
        folder,
        datafolder,
        sources=["tmdb", "letterboxd", "justwatch"],
        mode="naive",
    ):
        self.name = name
        self.folder = folder
        self.datafolder = datafolder
        self.sources = sources
        self.model_label = "deepseek-r1:1.5b"  # The name of the model for Ollama to download (all models here: https://ollama.com/search)
        self.outname = "deepseek1_5"
        self.chat_history = []
        self.mode = mode
        self.context = None
        self._download_model_if_missing()

        with open("./models/deepseek/prompt_template_deepseek.txt", "r") as fd:
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
        # return {"movies": ["challengers"], "people": []}

    def train(self):
        pass

    def reply(self, prompt):
        return self.prompt_nonstream(prompt)

    def prompt_stream(
        self, prompt: str, data: str = ""
    ) -> Iterator[ollama.GenerateResponse]:
        """Feeds the prompt to the model, returning its response as a stream iterator"""
        phrases = self.extract_keyphrases(prompt)
        s = Scraper(phrases, self.datafolder, self.outname, sources=self.sources)

        # TODO what should be the shape of data? currently I just concat things together
        if self.mode == "naive":
            for source in self.sources:
                context = open(s.files[source]).read()
                data += context
        elif self.mode == "advanced":
            summarizer = Summarizer()
            for source in self.sources:
                context = s.files[source]
                for key, item in phrases.items():
                    for i in item:
                        data += summarizer.summarize(context, i)
        elif self.mode == "modular":
            pass  # TODO
        else:  # this should never happen, but better safe than sorry
            data = ""

        # need to save this, so we can see it in the output file
        self.context = data

        final_prompt = self.prompt_template.format(data=data, query=prompt)
        return ollama.generate(model=self.model_label, prompt=final_prompt, stream=True)

    def prompt_nonstream(
        self, prompt: str, data: str = ""
    ) -> Tuple[ollama.GenerateResponse, str]:
        phrases = self.extract_keyphrases(prompt)
        s = Scraper(phrases, self.datafolder, self.outname, sources=self.sources)

        # TODO what should be the shape of data? currently I just concat things together
        if self.mode == "naive":
            for source in self.sources:
                context = open(s.files[source]).read()
                data += context
        elif self.mode == "advanced":
            summarizer = Summarizer()
            for source in self.sources:
                context = s.files[source]
                for key, item in phrases.items():
                    for i in item:
                        data += summarizer.summarize(context, i)
        elif self.mode == "modular":
            pass  # TODO
        else:  # this should never happen, but better safe than sorry
            data = ""

        # need to save this, so we can see it in the output file
        self.context = data

        """Feeds the prompt to the model, returning its response"""
        final_prompt = self.prompt_template.format(data=data, query=prompt)
        return (
            ollama.generate(model=self.model_label, prompt=final_prompt, stream=False),
            data,
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
