from scraper import *
from POStagger import *
from summarizer import *
import spacy

nlp = spacy.load("en_core_web_sm")

with open('./data/stopwords-en.txt', "r") as f:
    stop_words = f.readlines()
    stop_words = [word.strip() for word in stop_words]

class Rag():
    def __init__(self, prompt, mode, datafolder, outname, sources = ["tmdb", "letterboxd", "justwatch"], scraper=None):
        self.phrases = self.extract_keyphrases(prompt)
        print(self.phrases)
        if scraper: # probably irrelevant, just in case we ever wanna use a different scraper
            self.scraper = scraper
        else:
            self.scraper = Scraper(self.phrases, datafolder, outname, sources=sources)

        

        self.prompt = prompt
        self.mode = mode
        self.sources = sources

    # extract titles, people, ...?
    # return a dict of lists, data will be scraped for each element in each list
    def extract_keyphrases(self, prompt):
        tagger = POStagger()
        tagged = tagger.tag(prompt)

        tokens = nlp(prompt)
        tokens = [word.lemma_ for word in tokens if word.lemma_ not in stop_words]
        tagged["key"] = list(set(tokens))

        return tagged
    
    # returns context that goes into a model and state that goes into logger
    def get_context(self):
        data = ""
        state = {}
        
        #TODO what should be the shape of data? currently I just concat things together
        if self.mode == "naive":
            for source in self.sources:
                context = open(self.scraper.files[source], errors="ignore").read()
                data += context
        elif self.mode == "advanced":
            summarizer = Summarizer()
            state["summaries"] = []
            for source in self.sources:
                context = self.scraper.files[source]
                for key, item in self.phrases.items():
                    for i in item:
                        summary = summarizer.summarize(context, i)
                        data += summary
                        state["summaries"].append(summary)
        elif self.mode == "modular":
            pass  # TODO

        state["context"] = data
        return data, state

