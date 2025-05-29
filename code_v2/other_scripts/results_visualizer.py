import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Any
import jinja2
import json
import markdown


def get_gpt_metrics() -> List[List[str]]:
    metrics_name = ["Correctness", "Clarity", "AnswerRelevancy", "Faithfulness", "ContextualPrecision", "ContextualRecall", "ContextualRelevancy"]
    metrics = ["Correctness (GEval)", "Clarity (GEval)", "Answer Relevancy", "Faithfulness", "Contextual Precision", "Contextual Recall", "Contextual Relevancy"]
    folders = ["deepseek_advanced", "qwen_naive", "qwen_baseline", "deepseek_baseline", "deepseek_naive"]
    names = ["deepseek_advanced_evaluation", "qwen_naive_evaluation", "qwen_baseline_evaluation", "deepseek_baseline_evaluation", "deepseek_naive_evaluation"]
    base_dir = f"{os.getcwd()}/../../code/final_results_for_evaluation/GPT-evaluations"

    reasons = [None] * 50
    for name, folder in zip(names, folders):
        folder = f"{base_dir}/{folder}"
        for metric, met in zip(metrics, metrics_name):
            for file in os.scandir(folder):
                if name in file.name and met in file.name:
                    f = open(folder+"/"+file.name)
                    data = json.load(f)
                    for i in range(1, 51):
                        try:
                            reason = data["reasons"][f"test_case_{i}"][metric]
                            f = folder.split("/")[-1]
                            o = reasons[i - 1]
                            if o is not None:
                                if o.get(f) is not None:
                                    o[f][metric] = reason
                                else:
                                    o[f] = {
                                        metric: reason
                                    }
                            else:
                                reasons[i - 1] = {
                                    f: { metric: reason }
                                }
                            break
                        except Exception as e:
                            pass
    return reasons



def load_results(inputs: List[str]):
    question_answers = {}
    for input in inputs:
        name = input.split("/")[-1].replace(".json", "")
        with open(input, "r", encoding="utf-8") as f:
            file_content = json.load(f)
            for i, element in enumerate(file_content):
                if i > 25:
                    continue
                user_input = element.get("user_input")
                ground_truth = element.get("ground_truth")
                contexts = element.get("contexts")
                response = element.get("response")
                html_response = markdown.markdown(response)

                obj = question_answers.get(user_input)
                if obj:
                    obj[name] = {
                        "contexts": contexts,
                        "response": html_response
                    }
                    if obj.get("ground_truth", None) != ground_truth:
                        #print(f"ERROR: Ground truth for {name} (element {i}) is not in sync with previous!")
                        #print(f"     Expected {obj.get("ground_truth")}, got {ground_truth}")
                        obj[name]["gt_error"] = True
                    else:
                        obj[name]["gt_error"] = False
                else:
                    question_answers[user_input] = {
                        name: {
                            "contexts": contexts,
                            "response": html_response,
                            "gt_error": False
                        },
                        "ground_truth": ground_truth,
                        "i": i
                    }
    gpt_responses = get_gpt_metrics()
    for _, value in question_answers.items():
        i = value["i"]
        response = gpt_responses[i]
        if response is not None:
            for key, v in response.items():
                print(key)
                try:
                    value[key]["gpt"] = v
                except Exception as e:
                    pass
    return question_answers

def render_html(question_answers: Dict[str, Any], output: str):
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader("."),
        autoescape=False
    )
    template = env.get_template("template.html")
    html = template.render(questions_answers=question_answers)
    with open(output, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Visualizer")
    parser.add_argument("-i", "--inputs", nargs='+')
    parser.add_argument("-o", "--output")
    arguments = parser.parse_args()

    input_files: List[str] = arguments.inputs
    output_file: str = arguments.output

    qa = load_results(input_files)
    render_html(qa, output_file)