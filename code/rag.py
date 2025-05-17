from scraper import *
from POStagger import *
from summarizer import *
import spacy



class Rag():
    def __init__(self, prompt, mode, datafolder, outname, sources = ["tmdb", "letterboxd", "justwatch", "wiki"], scraper=None):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("en_core_web_sm load failed!")
            self.nlp = None

        print("Started looking for stopwords")
        with open('./data/stopwords-en.txt', "r") as f:
            self.stop_words = f.readlines()
            self.stop_words = [word.strip() for word in self.stop_words]
        self.phrases = self.extract_keyphrases(prompt)
        print(self.phrases)
        print("Finished looking for stopwords")
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
        print("Started POS tagger")
        tagger = POStagger()
        tagged = tagger.tag(prompt)

        if self.nlp:
            tokens = self.nlp(prompt)
            tokens = [word.lemma_ for word in tokens if word.lemma_ not in self.stop_words]
            tagged["key"] = list(set(tokens))
        else:
            tokens = []
            tagged["key"] = []
        print("Finished POS tagger") 	
        return tagged
    
    # returns context that goes into a model and state that goes into logger
    def get_context(self):
        data = ""
        state = {}
        
        #TODO what should be the shape of data? currently I just concat things together
        if self.mode == "naive":
            for key in self.scraper.files.keys():
                context = open(self.scraper.files[key], errors="ignore").read()
                data += context
        elif self.mode == "advanced":
            print("Started summarizing")
            summarizer = Summarizer()
            state["summaries"] = []
            for key in self.scraper.files.keys():
                context = self.scraper.files[key]
                for key, item in self.phrases.items():
                    for i in item:
                        summary = summarizer.summarize(context, i)
                        data += summary
                        state["summaries"].append(summary)
            print("Finished summarizing")

        state["context"] = data
        return data, state

