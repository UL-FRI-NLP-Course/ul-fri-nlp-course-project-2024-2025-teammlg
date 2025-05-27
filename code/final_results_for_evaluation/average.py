import os
import json

def average(name, metrics, dbfolder = "final_results_for_evaluation"):
    out_metrics = {}
    for metric in metrics:
        out_metrics[metric] = 0
        total = 0
        count = 0
        for file in os.scandir(dbfolder):
            if name in file.name:
                f = open(dbfolder+"/"+file.name)
                data = json.load(f)
                if metric in data:
                    total += data[metric]["average"]
                    count += 1
        
        if count > 0:
            out_metrics[metric] = total/count
        else:
            out_metrics[metric] = 0
    return out_metrics


if __name__ == "__main__":
    metrics = ["Correctness (GEval)", "Clarity (GEval)", "Answer Relevancy", "Faithfulness", "Contextual Precision", "Contextual Recall", "Contextual Relevancy"]
    names = ["qwen_naive_evaluation", "qwen_baseline_evaluation"]
    for name in names:
        print(name, average(name, metrics))
        