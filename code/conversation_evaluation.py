from models import *
import os
import datetime
import json
from memory import *

class ConversationEvaluation:
    def __init__(self, model):
        self.model = model

    def get_session_folder(self):
        now = str(datetime.datetime.now())
        now = now.replace(":", "_")
        now = now.replace(" ", "_")
        now = now.replace(".", "_")
        outfolder = "data/conversation_evaluation_data/" + str(now)
        if not os.path.isdir("data/conversation_evaluation_data/"):
            os.makedirs("data/conversation_evaluation_data/")
        os.makedirs(outfolder)
        return outfolder

    def evaluate(self):
        session_folder = self.get_session_folder()
        print("Evaluating", self.model.name, ":")
        replies = []
        prompt = input("> ").strip()
        prompts = [prompt]
        contexts = []
        evalout = []
        while prompt != "quit":
            fullresponse, state = self.model.prompt_stream(prompt, data="")
            print(fullresponse)

            evaldict = {"query": prompt, "reply": fullresponse}
            evaldict.update(state)
            evalout.append(evaldict)
            replies.append(fullresponse)
            contexts.append(str(state["context"]))
            prompt = input("> ").strip()
            if prompt != "quit":
                prompts.append(prompt)

        outf = self.writereplies(prompts, replies, self.model.folder)
        self.writeparams(outf)

        with open(session_folder + "/" + self.model.outname + "_" + self.model.mode + ".json", "a") as outfile:
                json.dump(state, outfile, indent=4)

    def writeparams(self, outfolder):
        outdict = {}
        outdict["mode"] = self.model.mode
        outdict["sources"] = self.model.sources
        outdict["context"] = self.model.context

        with open(outfolder+"/parameters.json", "w") as outfile:
            json.dump(outdict, outfile, indent=4)

    def writereplies(self, prompts, replies, folder):
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

        return outfolder

if __name__ == "__main__":
    qwen_name = "Qwen/Qwen3-8B"
    qwen_outname = "qwen3_8b"
    #deepseekbaseline = DeepSeekBaseline("deepseek-r1:1.5b-baseline", "models/deepseek_baseline", "data/scraped_data")
    #deepseek = DeepSeekFilmChatBot("deepseek-r1:1.5b", "models/deepseek", "data/scraped_data")
    #deepseekadvanced = DeepSeekFilmChatBot("deepseek-r1:1.5b", "models/deepseek", "data/scraped_data", mode="advanced")
    #qwenbaseline = QwenBaseline(qwen_name, "models/qwen_baseline", "data/scraped_data", qwen_outname)
    qwen = QwenChatBot(qwen_name, "models/qwen", "data/scraped_data", qwen_outname)
    qwenadvanced = QwenChatBot(qwen_name, "models/qwen", "data/scraped_data", qwen_outname, mode="advanced")

    #e = ConversationEvaluation(qwenadvanced)
    e = ConversationEvaluation(qwen)
    results = e.evaluate()
