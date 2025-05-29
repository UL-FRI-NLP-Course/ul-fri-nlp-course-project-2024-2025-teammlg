import argparse
from typing import Dict, List, Any
import jinja2
import json
import markdown


def load_results(inputs: List[str]):
    question_answers = {}
    for input in inputs:
        with open(input, "r", encoding="utf-8") as f:
            file_content = json.load(f)
            for i, element in enumerate(file_content):
                user_input = element.get("user_input")
                ground_truth = element.get("ground_truth")
                contexts = element.get("contexts")
                response = element.get("response")
                html_response = markdown.markdown(response)

                obj = question_answers.get(user_input)
                if obj:
                    obj[input] = {
                        "contexts": contexts,
                        "response": html_response
                    }
                    if obj.get("ground_truth", None) != ground_truth:
                        print(f"ERROR: Ground truth for {input} (element {i}) is not in sync with previous!")
                        print(f"     Expected {obj.get("ground_truth")}, got {ground_truth}")
                        obj[input]["gt_error"] = True
                    else:
                        obj[input]["gt_error"] = False
                else:
                    question_answers[user_input] = {
                        input: {
                            "contexts": contexts,
                            "response": html_response,
                            "gt_error": False
                        },
                        "ground_truth": ground_truth
                    }
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