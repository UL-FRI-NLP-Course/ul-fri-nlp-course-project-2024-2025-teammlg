from collections import namedtuple
import argparse
import json
from typing import Any, Dict, List, TypedDict


class AlternativeResult(TypedDict):
    user_prompt: str
    data: str
    generated_response: str


def crop_assistant_response(obj: AlternativeResult) -> str:
    response = obj["generated_response"].split("</think>")[-1]
    response = response.strip()
    return response


class AlternativeJson(TypedDict):
    model: str
    rag_type: str
    results: List[AlternativeResult]


class EvaluateResult:
    def __init__(self, user_input: str, contexts: str, response: str, ground_truth: str):
        self.user_input = user_input
        self.contexts = contexts
        self.response = response
        self.ground_truth = ground_truth


def convert_alternative_to_evaluate(input_file: str, output_file: str):
    print(f"Converting {input_file} to {output_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        alternative_json: AlternativeJson = json.load(f)
    
    with open("../../code/data/evaluation_questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)
    ground_truths = questions["ground_truth"]

    evaluate_json = []
    for result, gt in zip(alternative_json["results"], ground_truths):
        user_input = result["user_prompt"]
        contexts = result["data"]
        response = crop_assistant_response(result)
        ground_truth = gt[0]
        evaluate_json.append({
            "user_input": user_input,
            "contexts": contexts,
            "response": response,
            "ground_truth": ground_truth
        })
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(evaluate_json, f, ensure_ascii=False)
    
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Converter")
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    arguments = parser.parse_args()

    input_file: str = arguments.input
    output_file: str = arguments.output

    convert_alternative_to_evaluate(input_file, output_file)