import re
import os
import argparse
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import git
import sys
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
from models import *
import datetime
import json
import random
import time
from metrics import *

class Evaluation:
    # so the idea is: you can evaluate models on a specific movie you pass as an argument, or the evaluator will pick a random one from this list for each query
    # i picked these 4 for the following reasons:
    #  - challengers has a lot of meme reviews on top pages of letterboxd, I'm interested if the model can get anything useful out of that (twitter data would probably be very similar)
    #  - snow white, because there are different ones with the same title, would be interesting to know if the model gets confused
    #  - madame is athletic: some extremely old movie with almost no reviews
    #  - 28 years later: upcoming movie, if we're gonna be asking about release dates, etc. (it's also a sequel, so there might be some interesting ambiguity)
    def __init__(
        self,
        models,
        queryfile,
        moviename=["Challengers", "snow White", "Madame is Athletic", "28 years later"],
    ):
        self.models = models
        self.queryfile = queryfile
        self.queries = []
        f = open(queryfile)
        for line in f:
            # TODO maybe handle misspellings, etc. (that is, deliberately put them in and see if the model handles them correctly)
            if isinstance(moviename, str):
                self.queries.append(line.strip().replace("{title}", moviename))
            else:
                self.queries.append(
                    line.strip().replace(
                        "{title}", moviename[random.randint(0, len(moviename)) - 1]
                    )
                )

    def get_session_folder(self):
        now = str(datetime.datetime.now())
        now = now.replace(":", "_")
        now = now.replace(" ", "_")
        now = now.replace(".", "_")
        outfolder = "data/evaluation_data/" + str(now)
        if not os.path.isdir("data/evaluation_data/"):
            os.makedirs("data/evaluation_data/")
        os.makedirs(outfolder)
        return outfolder

    # for evaluation_questions.txt
    def evaluate(self, printout=False):
        # create folder for logging intermediate states (contexts, summarizations, etc.)
        # entire session will be logged in the same folder (so, each model in separate file)
        session_folder = self.get_session_folder()

        for model in self.models:
            print("Evaluating", model.name, model.mode, ":")
            results = []
            execution_times = []
            contexts = []
            for q in self.queries:
                if printout:
                    print("Query:", q, "\n")

                start = time.time()
                reply, state = model.reply(q)
                execution_times.append(time.time() - start)

                if printout:
                    print("Reply:", reply, "\n\n")

                results.append(str(reply))
                contexts.append(str(state["context"]))

            outf = self.write_replies(self.queries, results, model.folder)
            self.write_params(outf, model)
            self.write_results(outf, self.queries, results, contexts, execution_times)
            self.write_contexts(self.queries, contexts, outf)

            with open(
                session_folder + "/" + model.outname + "_" + model.mode + ".json", "a"
            ) as outfile:
                json.dump(state, outfile, indent=4)

    # for evaluation_questions.json (with ground truths)
    def evaluateGT(self, fileWithGT, printout=False):
        # create folder for logging intermediate states (contexts, summarizations, etc.)
        # entire session will be logged in the same folder (so, each model in separate file)
        session_folder = self.get_session_folder()

        for model in self.models:
            print("Evaluating", model.name, model.mode, ":")
            results = []
            execution_times = []
            contexts = []

            with open(fileWithGT, mode="r", encoding="UTF-8") as fwgt:
                data = json.load(fwgt)

            queries = []
            for query in data["scenarios"]:
                totalquery = ""
                for question in query:
                    totalquery += str(question) + " "
                queries.append(totalquery.strip())

            gts = []
            for gt in data["ground_truth"]:
                totalgt = ""
                for g in gt:
                    totalgt += str(g) + " "
                gts.append(totalgt.strip())

            # in case you don't wanna run on entire testset
            #queries = queries[:5]
            #gts = gts[:5]

            evalout = []
            for q, gt in zip(queries, gts):
                if printout:
                    print("Query:", q, "\n")

                start = time.time()
                reply, state = model.reply(q)

                # Removes the reasoning part
                reply = re.sub("<think>(.|\r|\n)*?</think>", "", reply)
                reply = re.sub("<｜User｜>(.|\r|\n)*?<｜Assistant｜>", "", reply)
                reply = reply.replace("system\nYou are an AI assistant tasked with helping the user on film or series-related questions. Read the following data and answer the question. If you cannot infer information from the data, do not answer the question.\nuser\n", "")

                execution_times.append(time.time() - start)

                evaldict = {"query": q, "reply": reply}
                evaldict.update(state)
                evalout.append(evaldict)

                if printout:
                    print("Reply:", reply, "\n\n")

                results.append(str(reply))
                contexts.append(state["context"])

            outf = self.write_replies(queries, results, model.folder)
            self.write_params(outf, model)
            self.write_results(outf, queries, results, contexts, execution_times, gts)
            self.write_contexts(queries, contexts, outf)
            self.write_combined(queries, results, contexts, gts, outf)

            with open(
                session_folder + "/" + model.outname + "_" + model.mode + ".json", "a"
            ) as outfile:
                json.dump(evalout, outfile, indent=4)

    def compute_similarities(self, queries, results, contexts):
        metrics = {}
        metrics["tf-idf_qr"] = []
        metrics["tf-idf_qc"] = []
        metrics["tf-idf_rc"] = []

        # comparing replies against queries
        for q, r in zip(queries, results):
            metrics["tf-idf_qr"].append(tf_idf_qr(q, results))

        # comparing contexts (i.e. scraped data) against queries
        for q, c in zip(queries, contexts):
            metrics["tf-idf_qc"].append(tf_idf_qc(q, c))

        # comparing replies against contexts
        for r, c in zip(results, contexts):
            metrics["tf-idf_rc"].append(tf_idf_rc(r, c))

        return metrics

    def compute_GT_metrics(self, replies, gts):
        metrics = {}
        metrics["rougeL"] = rougeL(replies, gts)
        metrics["bleu"] = bleu(replies, gts)

        return metrics

    def write_results(self, outfolder, queries, results, contexts, times, gts=None):
        outdict = {}
        outdict["execution times"] = times
        outdict["avg execution time"] = sum(times) / len(times)

        metrics = self.compute_similarities(queries, results, contexts)
        for key, val in metrics.items():
            outdict[key] = val
            # outdict["avg "+key] = sum(val) / len(val)

        # if we have ground truths we can also compute ROUGE, BLEU, etc.
        if gts:
            gt_metrics = self.compute_GT_metrics(results, gts)
            outdict.update(gt_metrics)

        with open(outfolder + "/results.json", "w") as outfile:
            json.dump(outdict, outfile, indent=4)

    def write_params(self, outfolder, model):
        outdict = {}
        outdict["mode"] = model.mode
        outdict["sources"] = model.sources

        with open(outfolder + "/parameters.json", "w") as outfile:
            json.dump(outdict, outfile, indent=4)

    def write_replies(self, queries, results, folder):
        now = str(datetime.datetime.now())
        now = now.replace(":", "_")
        now = now.replace(" ", "_")
        now = now.replace(".", "_")
        outfolder = folder + "/results/" + str(now)
        if not os.path.isdir(folder + "/results"):
            os.makedirs(folder + "/results")
        os.makedirs(outfolder)

        out = [{"query": q, "reply": r} for q, r in zip(queries, results)]

        with open(outfolder + "/replies.json", "w") as outfile:
            json.dump(out, outfile, indent=4)

        return outfolder
    
    def write_contexts(self, queries, contexts, outfolder):
        out = [{"query": q, "context": r} for q, r in zip(queries, contexts)]

        with open(outfolder + "/contexts.json", "w") as outfile:
            json.dump(out, outfile, indent=4)

        return outfolder
    
    def write_combined(self, queries, replies, contexts, gts, outfolder):
        out = [{"user_input": q, "contexts": c, "response": r, "ground_truth": g} for q, c, r, g in zip(queries, contexts, replies, gts)]

        with open(outfolder + "/combined_data_for_evaluation.json", "w") as outfile:
            json.dump(out, outfile, indent=4)

        return outfolder


if __name__ == "__main__":
    deepseek_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    deepseek_outname = "deepseek_r1_8b"
    qwen_name = "Qwen/Qwen3-8B"
    qwen_outname = "qwen3_8b"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Specify which models to use (comma separated (no space), if you want multiple). Options: deepseek_baseline, deepseek_naive, deepseek_advanced, qwen_baseline, qwen_naive, qwen_advanced.", required=True, action='store')
    args = parser.parse_args()
    chosen_models = str(args.model).split(",")

    models = []
    with suppress_stdout():
        if "deepseek_baseline" in chosen_models:
            models.append(DeepSeekBaseline(deepseek_name, "models/deepseek_baseline", "data/scraped_data", deepseek_outname+"_baseline"))
        if "deepseek_naive" in chosen_models:
            models.append(DeepSeekFilmChatBot(deepseek_name, "models/deepseek", "data/scraped_data", deepseek_outname+"_naive"))
        if "deepseek_advanced" in chosen_models:
            models.append(DeepSeekFilmChatBot(deepseek_name, "models/deepseek", "data/scraped_data", deepseek_outname+"_advanced", mode="advanced"))
        if "qwen_baseline" in chosen_models:
            models.append(QwenBaseline(qwen_name, "models/qwen_baseline", "data/scraped_data", qwen_outname+"_baseline"))
        if "qwen_naive" in chosen_models:
            models.append(QwenChatBot(qwen_name, "models/qwen", "data/scraped_data", qwen_outname+"_naive"))
        if "qwen_advanced" in chosen_models:
            models.append(QwenChatBot(qwen_name, "models/qwen", "data/scraped_data", qwen_outname+"_advanced", mode="advanced"))
    
    e = Evaluation(models, "data/evaluation_questions.txt")
    #results = e.evaluate()
    gteval = e.evaluateGT("data/evaluation_questions.json", printout=True)
