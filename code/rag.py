from abc import ABC, abstractmethod
import logging
from scraper import *
from POStagger import *
from summarizer import *
import spacy
from llmlingua import PromptCompressor


class SimpleRAG:
    def __init__(self, label: str = "SimpleRAG", accurate: bool = False, logging_level: int = logging.INFO, log_to_console: bool = False):
        self.logger = logging.getLogger(label)
        self.logger.setLevel(logging_level)
        self.logger.addHandler(logging.StreamHandler())
        if not accurate:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("Loaded spacy model (en_core_web_sm)")
            except Exception as e:
                self.logger.error(f"Could not load spacy model ('en_core_web_sm'): {e}")
                self.nlp = None
        else:
            try:
                self.nlp = spacy.load("en_core_web_trf")
                self.logger.info("Loaded spacy model ('en_core_web_trf')")
            except Exception as e:
                self.logger.error(f"Could not load spacy model ('en_core_web_trf'): {e}")
                self.nlp = None


class Rag():
    def __init__(self, mode, datafolder, outname, sources = ["tmdb", "letterboxd", "justwatch", "wiki"], scraper=None):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("en_core_web_sm load failed!")
            self.nlp = None

        with open('./data/stopwords-en.txt', "r") as f:
            self.stop_words = f.readlines()
            self.stop_words = [word.strip() for word in self.stop_words]

        if scraper: # probably irrelevant, just in case we ever wanna use a different scraper
            self.scraper = scraper
        else:
            self.scraper = Scraper(datafolder, outname, sources=sources)

        self.mode = mode
        self.sources = sources
        self.summarizer = Summarizer()

    # extract titles, people, ...?
    # return a dict of lists, data will be scraped for each element in each list
    def extract_keyphrases(self, prompt):
        tagger = POStagger()
        tagged = tagger.tag(prompt)

        if self.nlp:
            tokens = self.nlp(prompt)
            tokens = [word.lemma_ for word in tokens if word.lemma_ not in self.stop_words]
            tagged["key"] = list(set(tokens))
        else:
            tokens = []
            tagged["key"] = []	
        return tagged
    
    # returns context that goes into a model and state that goes into logger
    def get_context(self, prompt: str):
        data = ""
        state = {}
        
        phrases = self.extract_keyphrases(prompt)
        files = self.scraper.run(phrases)
        
        #TODO what should be the shape of data? currently I just concat things together
        if self.mode == "naive":
            for key in files.keys():
                context = open(files[key], errors="ignore").read()
                data += context
        elif self.mode == "advanced":
            state["summaries"] = []
            for key in files.keys():
                context = open(files[key], errors="ignore").read()
                summary = self.summarizer.extract_important(context, prompt)
                data += summary
                data += "\n"
                state["summaries"].append(summary)

        state["context"] = data
        return data, state


if __name__ == "__main__":
    prompt = input("> ")
    rag = Rag(mode="advanced", datafolder="test_rag_data", outname="test_rag")

    while prompt != "quit":
        data, state = rag.get_context(prompt)
        print(data)
        print()
        prompt = input("> ")