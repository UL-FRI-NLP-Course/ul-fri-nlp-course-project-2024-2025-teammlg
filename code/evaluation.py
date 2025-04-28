from models import *
import os
import datetime
import json
import random

class Evaluation:
    #so the idea is: you can evaluate models on a specific movie you pass as an argument, or the evaluator will pick a random one from this list for each query
    #i picked these 4 for the following reasons: 
    #  - challengers has a lot of meme reviews on top pages of letterboxd, I'm interested if the model can get anything useful out of that (twitter data would probably be very similar)
    #  - snow white, because there are different ones with the same title, would be interesting to know if the model gets confused
    #  - madame is athletic: some extremely old movie with almost no reviews
    #  - 28 years later: upcoming movie, if we're gonna be asking about release dates, etc. (it's also a sequel, so there might be some interesting ambiguity)
    def __init__(self, models, queryfile, moviename=["Challengers", "snow White", "Madame is Athletic", "28 years later"]):
        self.models = models
        self.queryfile = queryfile
        self.queries = []
        f = open(queryfile)
        for line in f:
            #TODO maybe handle misspellings, etc. (that is, deliberately put them in and see if the model handles them correctly)
            if isinstance(moviename, str):
                self.queries.append(line.strip().replace("{title}", moviename))
            else:
                self.queries.append(line.strip().replace("{title}", moviename[random.randint(0, len(moviename))-1]))

    def evaluate(self, printout=False):
        for model in self.models:
            print("Evaluating", model.name, ":")
            result = []
            for q in self.queries:
                if printout:
                    print("Query:", q, "\n")
                reply = model.reply(q).response
                if printout:
                    print("Reply:", reply, "\n\n")
                result.append(str(reply))
            outf = self.writeresults(result, model.folder)
            self.writeparams(outf, model)

    def writeparams(self, outfolder, model):
        outdict = {}
        outdict["mode"] = model.mode
        outdict["sources"] = model.sources
        outdict["context"] = model.context

        with open(outfolder+"/parameters.json", "w") as outfile:
            json.dump(outdict, outfile, indent=4)


    def writeresults(self, results, folder):
        now = str(datetime.datetime.now())
        now = now.replace(":", "_")
        now = now.replace(" ", "_")
        now = now.replace(".", "_")
        outfolder = folder+"/results/"+str(now)
        if not os.path.isdir(folder+"/results"):
            os.makedirs(folder+"/results")
        os.makedirs(outfolder)

        #TODO maybe report some other stats - execution times, etc.

        out = [{'query': q,
            'reply': r
            } for q, r in zip(self.queries, results)]
        
        with open(outfolder+"/results.json", "w") as outfile:
            json.dump(out, outfile, indent=4)

        return outfolder

if __name__ == "__main__":
    deepseekbaseline = DeepSeekBaseline("deepseek-r1:1.5b-baseline", "models/deepseek_baseline", "data/scraped_data")
    deepseek = DeepSeekFilmChatBot("deepseek-r1:1.5b", "models/deepseek", "data/scraped_data")
    deepseekadvanced = DeepSeekFilmChatBot("deepseek-r1:1.5b", "models/deepseek", "data/scraped_data", mode="advanced")
    qwenbaseline = QwenBaseline("qwen:1.8b", "models/qwen_baseline", "data/scraped_data")
    qwen = QwenChatBot("qwen:1.8b", "models/qwen", "data/scraped_data")
    qwenadvanced = QwenChatBot("qwen:1.8b", "models/qwen", "data/scraped_data", mode="advanced")

    #models = [deepseekbaseline, deepseek] # add new models here
    models = [deepseekbaseline, deepseek, deepseekadvanced, qwenbaseline, qwen, qwenadvanced]
    e = Evaluation(models, "data/evaluation_questions.txt")
    results = e.evaluate()
