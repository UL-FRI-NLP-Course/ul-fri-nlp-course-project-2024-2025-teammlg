import json
import os
import sys
import uuid
from pipeline import ChatbotPipeline


def evaluate(pipeline: ChatbotPipeline, output_directory: str):
    print("Start evaluation...")
    with open("data/evaluation_questions.json", "r") as f:
        content = f.read()
        json_content = json.loads(content)
    
    os.makedirs(output_directory, exist_ok=True)
    results_file = f"{output_directory}/{uuid.uuid4().hex}.json"
    results = {
        "model": pipeline.model.model_label,
        "rag_type": str(pipeline.rag_type),
        "results": []
    }
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    print("Everything prepared, starting...")

    scenarios = json_content["scenarios"]
    n = len(scenarios)
    for i, scenario in enumerate(scenarios):
        print(f"Testing prompt {i}/{n}...")
        try:
            prompt = scenario[0]
            result = pipeline.run(prompt)
            results["results"].append(result)
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)
        except Exception as e:
            print(e, file=sys.stderr)

    print("Done!")