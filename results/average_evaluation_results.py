import os
import re
import json
import sys
from collections import defaultdict

directory = ""
if len(sys.argv) == 2:
    directory = sys.argv[1]
else:
    raise Exception("Only one argument is optional - directory path.")

folders = ["deepseek_baseline",
           "deepseek_naive",
           "deepseek_advanced",
           "qwen_baseline",
           "qwen_naive",
           "qwen_advanced"]

for folder in folders:
    path = os.path.join(directory, folder)
    file_paths = os.listdir(path)
    """
    with open(f"{folder}.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    contexts = dict()
    for i, d in enumerate(data, start=1):
        contexts[f"test_case_{i}"] = d["contexts"]
    """
    counts = defaultdict(int)
    averages = defaultdict(float)
    evaluation_results = defaultdict(float)
    reasons = defaultdict(lambda: defaultdict(list))

    for file in file_paths:
        file_path = os.path.join(path, file)

        with open(file_path, "r") as file:
            x = json.load(file)
            for key, value in x.items():
                if type(key) == str and type(value) == str:
                    # key: "model_for_evaluation"
                    # value: str
                    evaluation_results[key] = value
                if type(key) == str and type(value) == float:
                    # key: "evaluation_time"
                    # value: float
                    evaluation_results[key] += value
                if type(key) == str and type(value) == dict and key == "reasons":
                    # key: "reasons"
                    # value: dict <metric_name, list>
                    for test in value:
                        for k, v in x[key][test].items():
                            reasons[test][k].extend(v)
                elif type(key) == str and type(value) == dict:
                    # key: str (metric name)
                    # value: dict <str, float>
                    if key in ["Faithfulness", "ContextualPrecision", "ContextualRecall", "ContextualRelevancy"] and value["average"] > 0:
                        counts[key] += 1
                        averages[key] += value["average"]
                    else:
                        counts[key] += 1
                        averages[key] += value["average"]

    evaluation_results["averages"] = {key: avg/counts[key] for key, avg in averages.items()}
    evaluation_results["reasons"] = dict(sorted(reasons.items(), key=lambda item: int(item[0].split("_")[2])))
    
    with open(os.path.join(directory, folder + f".json"), "w") as file:
        json.dump(evaluation_results, file, indent=4)
