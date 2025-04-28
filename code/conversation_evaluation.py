from models import *
import os
import datetime
import json
import ollama
from typing import Iterator

class ConversationEvaluation:
    def __init__(self, model):
        self.model = model

    def evaluate(self):
        print("Evaluating", self.model.name, ":")
        replies = []
        prompt = input("> ").strip()
        prompts = [prompt]
        while prompt != "quit":
            responses: Iterator[ollama.GenerateResponse] = self.model.prompt_stream(prompt, data="")
            thinking = True
            fullresponse = ""
            for i, response in enumerate(responses):
                if not thinking:
                    print(response.response, end="", flush=True)
                    fullresponse += response.response
                if response.response == "</think>":
                    thinking = False
            print()
            replies.append(fullresponse)
            prompt = input("> ").strip()
            if prompt != "quit":
                prompts.append(prompt)

        self.writeresults(prompts, replies, self.model.folder)

    def writeresults(self, prompts, replies, folder):
        now = str(datetime.datetime.now())
        now = now.replace(":", "_")
        now = now.replace(" ", "_")
        now = now.replace(".", "_")
        outfolder = folder+"/conversation_logs/"+str(now)
        if not os.path.isdir(folder+"/conversation_logs"):
            os.makedirs(folder+"/conversation_logs")
        os.makedirs(outfolder)

        #TODO maybe report some other stats - execution times, etc.

        out = [{'query'+str(i): q,
            'reply'+str(i): r
            } for i, (q, r) in enumerate(zip(prompts, replies))]
        
        with open(outfolder+"/conversation.json", "w") as outfile:
            json.dump(out, outfile, indent=4)

if __name__ == "__main__":
    deepseek = DeepSeekFilmChatBot("deepseek-r1:1.5b", "models/deepseek", "data/scraped_data")
    e = ConversationEvaluation(deepseek)
    #deepseekbaseline = DeepSeekBaseline("deepseek-r1:1.5b-baseline", "models/deepseek_baseline", "data/scraped_data")
    #e = ConversationEvaluation(deepseekbaseline)
    results = e.evaluate()
