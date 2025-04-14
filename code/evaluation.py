from models import DeepSeekFilmChatBot
import os
import datetime
import json

class Evaluation:
    def __init__(self, models, queryfile):
        self.models = models
        self.queryfile = queryfile
        self.queries = []
        f = open(queryfile)
        for line in f:
            #TODO lines in evaluation_questions.txt contain {title} as a placeholder
            #we need to decide whether to replace it with a random movie title each time, or just handpick a few meaningful ones
            #also, while we're at it, maybe handle misspellings, etc. (that is, deliberately put them in and see if the model handles them correctly)
            self.queries.append(line.strip()) 

    def evaluate(self):
        for model in self.models:
            print("Evaluating", model.name, ":")
            result = []
            for q in self.queries:
                print("Query:", q, "\n")
                reply = model.reply(q).response
                print("Reply:", reply, "\n\n")
                result.append(str(reply))
            self.writeresults(result, model.folder)

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

if __name__ == "__main__":
    deepseek = DeepSeekFilmChatBot("deepseek-r1:1.5b", "deepseek")
    models = [deepseek] # add new models here
    e = Evaluation(models, "data/evaluation_questions.txt")
    results = e.evaluate()
