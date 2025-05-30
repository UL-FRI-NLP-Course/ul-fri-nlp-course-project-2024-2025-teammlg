import os
import json

def average(name, folder, metrics, metrics_n):
    out_metrics = {}
    for metric, met in zip(metrics, metrics_n):
        out_metrics[metric] = 0
        total = 0
        count = 0

        for file in os.scandir(folder):
            if name in file.name and met in file.name:
                f = open(folder+"/"+file.name)
                data = json.load(f)
                if metric in data:
                    # ignore cases where there's no context (only relevant for advanced, because it can choose not to use rag)
                    if "Context" in metric and "advanced" in name and data[metric]["average"] == 0:
                        continue

                    total += data[metric]["average"]
                    count += 1
        
        if count > 0:
            out_metrics[metric] = total/count
        else:
            out_metrics[metric] = 0
    return out_metrics


def average2(name, metrics, folder):
    out_metrics = {}
    for metric in metrics:
        out_metrics[metric] = 0
        total = 0
        count = 0
        for file in os.scandir(folder):
            if name in file.name:
                f = open(folder+"/"+file.name)
                data = json.load(f)
                #if "Context" in metric and data[metric]["average"] == 0:
                if metric in data:
                    if "Context" in metric and "advanced" in name and data[metric]["average"] == 0:
                        continue
                # ignore cases where there's no context (only relevant for advanced, because it can choose not to use rag)
                #if "Context" in metric and "advanced" in name and data[metric]["average"] == 0:
                    total += data[metric]["average"]
                    count += 1
        
        if count > 0:
            out_metrics[metric] = total/count
        else:
            out_metrics[metric] = 0
    return out_metrics

def average3(name, folder, metrics, metrics_n):
    out_metrics = {}
    for metric, met in zip(metrics, metrics_n):
        out_metrics[metric] = 0
        total = 0
        count = 0

        for file in os.scandir(folder):
            if name in file.name and met in file.name:
                f = open(folder+"/"+file.name)
                data = json.load(f)
                if metric in data:
                    # ignore cases where there's no context (only relevant for advanced, because it can choose not to use rag)
                    if "Context" in metric and "advanced" in name and data[metric]["average"] == 0:
                        continue

                    total += data[metric]["average"]
                    count += 1
        
        if count > 0:
            out_metrics[metric] = total/count
        else:
            out_metrics[metric] = 0
    return out_metrics

if __name__ == "__main__":
    metrics_name = ["Correctness", "Clarity", "AnswerRelevancy", "Faithfulness", "ContextualPrecision", "ContextualRecall", "ContextualRelevancy"]
    metrics = ["Correctness (GEval)", "Clarity (GEval)", "Answer Relevancy", "Faithfulness", "Contextual Precision", "Contextual Recall", "Contextual Relevancy"]
    folders = ["deepseek_advanced"]
    names = ["deepseek_advanced_evaluation"]
    results = {}

    for name, folder in zip(names, folders):
        results["deepseek_advanced"] = average(name, folder, metrics, metrics_name)

    metrics = ["Correctness (GEval)", "Clarity (GEval)", "Answer Relevancy", "Faithfulness", "Contextual Precision", "Contextual Recall", "Contextual Relevancy"]
    names = ["qwen_baseline_evaluation", "qwen_naive_evaluation", "qwen_advanced_evaluation", "deepseek_baseline_evaluation", "deepseek_naive_evaluation"]
    n = ["qwen_baseline", "qwen_naive", "qwen_advanced", "deepseek_baseline", "deepseek_naive"]
    for name, actual_name in zip(names, n):
        results[name] = average2(name, metrics, actual_name)

    with open("results.json", "w") as outfile:
        json.dump(results, outfile, indent=4)


        