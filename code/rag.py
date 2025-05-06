from scraper import *
from POStagger import *
from summarizer import *
import nltk

class Rag():
    def __init__(self, prompt, mode, datafolder, outname, sources = ["tmdb", "letterboxd", "justwatch"], scraper=None):
        self.phrases = self.extract_keyphrases(prompt)
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

