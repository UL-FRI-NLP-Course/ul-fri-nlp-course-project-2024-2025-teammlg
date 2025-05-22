print("Loading models (could take up to 30 minutes)...")
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git
import sys
import argparse
from contextlib import contextmanager
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
with suppress_stdout():
    from models import *
    import os
    import re
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
            fullresponse = re.sub("<think>(.|\r|\n)*?</think>", "", fullresponse)
            fullresponse = re.sub("<｜User｜>(.|\r|\n)*?<｜Assistant｜>", "", fullresponse)
            fullresponse.replace("system\nYou are an AI assistant tasked with helping the user on film or series-related questions. Read the following data and answer the question. If you cannot infer information from the data, do not answer the question.\nuser\n", "")
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

        out = [{'query'+str(i): q,
            'reply'+str(i): r
            } for i, (q, r) in enumerate(zip(prompts, replies))]
        
        with open(outfolder+"/conversation.json", "w") as outfile:
            json.dump(out, outfile, indent=4)

        return outfolder

if __name__ == "__main__":
    deepseek_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    deepseek_outname = "deepseek_r1_8b"
    qwen_name = "Qwen/Qwen3-8B"
    qwen_outname = "qwen3_8b"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Specify which model to use. Options: deepseek_baseline, deepseek_naive, deepseek_advanced, qwen_baseline, qwen_naive, qwen_advanced.", required=True, action='store')
    args = parser.parse_args()
    with suppress_stdout():
        if args.model == "deepseek_baseline":
            chosen_model = DeepSeekBaseline(deepseek_name, "models/deepseek_baseline", "data/scraped_data", deepseek_outname+"_baseline")
        elif args.model == "deepseek_naive":
            chosen_model = DeepSeekFilmChatBot(deepseek_name, "models/deepseek", "data/scraped_data", deepseek_outname+"_naive")
        elif args.model == "deepseek_advanced":
            chosen_model = DeepSeekFilmChatBot(deepseek_name, "models/deepseek", "data/scraped_data", deepseek_outname+"_advanced", mode="advanced")
        elif args.model == "qwen_baseline":
            chosen_model = QwenBaseline(qwen_name, "models/qwen_baseline", "data/scraped_data", qwen_outname+"_baseline")
        elif args.model == "qwen_naive":
            chosen_model = QwenChatBot(qwen_name, "models/qwen", "data/scraped_data", qwen_outname+"_naive")
        elif args.model == "qwen_advanced":
            chosen_model = QwenChatBot(qwen_name, "models/qwen", "data/scraped_data", qwen_outname+"_advanced", mode="advanced")
        else:
            raise "Wrong model name, please try again. Options: deepseek_baseline, deepseek_naive, deepseek_advanced, qwen_baseline, qwen_naive, qwen_advanced."

    e = ConversationEvaluation(chosen_model)
    results = e.evaluate()
