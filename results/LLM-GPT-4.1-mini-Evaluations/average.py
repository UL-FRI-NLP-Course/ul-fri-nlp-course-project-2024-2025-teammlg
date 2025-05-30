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


if __name__ == "__main__":
    results = {}
    metrics_name = ["Correctness", "Clarity", "AnswerRelevancy", "Faithfulness", "ContextualPrecision", "ContextualRecall", "ContextualRelevancy"]
    metrics = ["Correctness (GEval)", "Clarity (GEval)", "Answer Relevancy", "Faithfulness", "Contextual Precision", "Contextual Recall", "Contextual Relevancy"]
    folders = ["qwen_advanced", "deepseek_advanced", "qwen_naive", "qwen_baseline", "deepseek_baseline", "deepseek_naive"]
    names = ["qwen_advanced_evaluation", "deepseek_advanced_evaluation", "qwen_naive_evaluation", "qwen_baseline_evaluation", "deepseek_baseline_evaluation", "deepseek_naive_evaluation"]
    for name, folder in zip(names, folders):
        results[folder] = average(name, folder, metrics, metrics_name)

    with open("results.json", "w") as outfile:
        json.dump(results, outfile, indent=4)
