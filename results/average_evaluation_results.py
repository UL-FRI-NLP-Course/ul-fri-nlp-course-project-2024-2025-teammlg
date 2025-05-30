import os
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

    counts = defaultdict(int)
    averages = defaultdict(float)
    results = defaultdict(float)
    reasons = defaultdict(lambda: defaultdict(list))

    for file in file_paths:
        file_path = os.path.join(path, file)

        with open(file_path, "r") as file:
            x = json.load(file)
            for key, value in x.items():
                if type(key) == str and type(value) == str:
                    # key: "model_for_evaluation"
                    # value: str
                    results[key] = value
                if type(key) == str and type(value) == float:
                    # key: "evaluation_time"
                    # value: float
                    results[key] += value
                if type(key) == str and type(value) == dict and key == "reasons":
                    # key: "reasons"
                    # value: dict <metric_name, list>
                    for test in value:
                        for k, v in x[key][test].items():
                            reasons[test][k].extend(v)
                elif type(key) == str and type(value) == dict:
                    # key: str (metric name)
                    # value: dict <str, float>
                    counts[key] += 1
                    averages[key] += value["average"]

    results["averages"] = {key: avg/counts[key] for key, avg in averages.items()}
    results["reasons"] = dict(sorted(reasons.items(), key=lambda item: int(item[0].split("_")[2])))
    
    with open(os.path.join(directory, folder + f".json"), "w") as file:
        json.dump(results, file, indent=4)
