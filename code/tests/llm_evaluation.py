# meta-llama/Llama-3.1-8B-Instruct
# meta-llama/Llama-3.1-70B-Instruct

# meta-llama/Llama-3.3-70B-Instruct

# 01-ai/Yi-34B-Chat
# Qwen/Qwen3-14B

# Arguments:
# 1. model to evaluate (without json)
# 2. model used to evaluate (from huggingface)
# 3. [OPTIONAL] evaluate start test
# 4. [OPTIONAL] evaluate end test (inclusive [start, end])

import sys

if len(sys.argv) - 1 == 4:
    model_to_evaluate = sys.argv[1]
    model_name = sys.argv[2]
    start = int(sys.argv[3])
    end = int(sys.argv[4])
    evaluate_range = (start, end)
elif len(sys.argv) - 1 == 2:
    model_to_evaluate = sys.argv[1]
    model_name = sys.argv[2]
    evaluate_range = None
else:
    raise Exception("Not enough arguments")

print("start load libraries")
from time import time

print("time imported")

start = time()
import torch

print(f"torch loaded: {time() - start}s")

start = time()
import json_repair

print(f"json_repair loaded: {time() - start}s")

start = time()
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, pipeline

print(f"transformers loaded: {time() - start}s")

start = time()
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import GEval, AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, \
    ContextualRecallMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from deepeval import evaluate
from deepeval.evaluate import DisplayConfig

print(f"deepeval loaded: {time() - start}s")

start = time()
from accelerate import Accelerator

print(f"accelerate loaded: {time() - start}s")

start = time()
from pydantic import BaseModel

print(f"pydantic loaded: {time() - start}s")

start = time()
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)

print(f"lmformatenforcer loaded: {time() - start}s")

start = time()
import json

print(f"json loaded: {time() - start}s")

start = time()
import os

print(f"os loaded: {time() - start}s")

start = time()
from collections import defaultdict
import numpy as np
import warnings

print(f"other loaded: {time() - start}s")


# https://www.deepeval.com/guides/guides-using-custom-llms
class EvaluationModel(DeepEvalBaseLLM):
    def __init__(self, model_name, hugging_face_token, cache_dir="/d/hpc/projects/onj_fri/teammlg/models", *args,
                 **kwargs):
        # deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
        # deepseek-ai/DeepSeek-R1-Distill-Llama-8B
        # meta-llama/Meta-Llama-3-70B-Instruct
        # meta-llama/Meta-Llama-3-8B-Instruct
        # meta-llama/Meta-Llama-3.1-8B-Instruct

        self.model_label = model_name

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        self.accelerator = Accelerator()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_label,
            cache_dir=cache_dir,
            token=hugging_face_token
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_label,
            torch_dtype="bfloat16",
            cache_dir=cache_dir,
            device_map="auto",
            token=hugging_face_token,
            # quantization_config=quantization_config
        )
        # self.model = self.accelerator.prepare(self.model)

        super().__init__(model_name=model_name, *args, **kwargs)

        self.pad_token_id = self.tokenizer.eos_token_id

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel):
        prompt = prompt[
                 :-6] + "Provide your `statements` and `reason` in a concise and straightforward manner, no longer than few sentences.\nJSON:"
        max_new_tokens = 4096
        # print(prompt)

        # Create parser required for JSON confinement using lmformatenforcer
        parser = JsonSchemaParser(schema.model_json_schema())
        prefix_function = build_transformers_prefix_allowed_tokens_fn(
            self.tokenizer, parser
        )

        generator = pipeline(
            "text-generation",
            model=self.load_model(),
            tokenizer=self.tokenizer,
            use_cache=True,
            truncation=True,
            device_map="auto",
            max_new_tokens=max_new_tokens,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id
        )

        """inputs = self.tokenizer(prompt, return_tensors="pt").to(
            self.model.device
        )

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=prefix_function
        )

        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        output = decoded[len(prompt):]"""

        # Additional parameters to consider:
        # temperature
        # top_k
        # top_p
        # repetition_penalty

        # Output and load valid JSON
        output_dict = generator(prompt, prefix_allowed_tokens_fn=prefix_function)
        output = output_dict[0]["generated_text"][len(prompt):]
        # print("Raw output:", output, flush=True)

        try:
            json_result = json_repair.loads(output)
        except json.JSONDecodeError as e:
            # print("Error output:", repr(output), flush=True)
            # print("Error message:", e)
            pass

        # Return valid JSON object according to the schema DeepEval supplied
        return schema(**json_result)

    async def a_generate(self, prompt: str, schema: BaseModel):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model_label


class EvaluationWithLLM:
    def __init__(self, model_name_for_evaluation=None, include_reason=False, verbose_mode=False,
                 hugging_face_token=None):
        self.verbose_mode = verbose_mode
        self.model_name = model_name_for_evaluation

        self.print(f"Cuda devices:")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                self.print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            self.print("CUDA is not available!")

        if model_name_for_evaluation is not None:
            self.print(f"Initializing model {self.model_name}")
            start = time()
            self.evaluation_model = EvaluationModel(self.model_name, hugging_face_token=hugging_face_token)
            self.print(f"Model loaded: {time() - start}s")
            self.evaluation_model_name = self.evaluation_model.get_model_name()
        else:
            self.evaluation_model = "gpt-4.1-mini"
            self.evaluation_model_name = self.evaluation_model

        self.metrics = self.get_default_metrics(self.evaluation_model, include_reason=include_reason)

    def print(self, message):
        if self.verbose_mode:
            print(message)

    def get_default_metrics(self, evaluation_model, include_reason=False):
        return [
            GEval(
                name="Correctness",
                model=evaluation_model,
                evaluation_steps=[
                    "Check whether the facts in 'actual output' contradict any facts in 'expected output'",
                    "Ommited details are acceptable, as long as they are not too frequent"
                ],
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            ),
            GEval(
                name="Clarity",
                model=evaluation_model,
                evaluation_steps=[
                    "Evaluate whether the response uses clear and direct language.",
                    "Identify any vague or confusing parts that reduce understanding.",
                    "Identify any unnecessary repetition of information in the response."
                ],
                evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            ),
            AnswerRelevancyMetric(
                threshold=0.5,
                model=evaluation_model,
                include_reason=include_reason
            ),
            FaithfulnessMetric(
                threshold=0.5,
                model=evaluation_model,
                include_reason=include_reason
            ),
            ContextualPrecisionMetric(
                threshold=0.5,
                model=evaluation_model,
                include_reason=include_reason
            ),
            ContextualRecallMetric(
                threshold=0.5,
                model=evaluation_model,
                include_reason=include_reason
            ),
            ContextualRelevancyMetric(
                threshold=0.5,
                model=evaluation_model,
                include_reason=include_reason
            )
        ]

    def get_test_cases(self, json_data):
        test_cases = []

        for i, line in enumerate(json_data, start=1):
            test_cases.append(
                LLMTestCase(
                    name=f"test_case_{i}",
                    input=line["user_input"],
                    actual_output=line["response"],
                    expected_output=line["ground_truth"],
                    retrieval_context=[line["contexts"]]
                )
            )

        return test_cases

    def load_json(self, results_file_path):
        with open(results_file_path) as file:
            data = json.load(file)

        self.print(f"Getting results from {results_file_path} is successful")

        return data

    def evaluate_results(self, results_file_path, evaluation_output_path, evaluate_range=None):
        start = time()

        with open(results_file_path) as file:
            json_data = json.load(file)

        test_cases = self.get_test_cases(json_data)

        self.print(f"{len(test_cases)} test cases loaded: {time() - start}s")

        if evaluate_range is not None:
            test_cases = test_cases[evaluate_range[0] - 1: evaluate_range[1]]
            self.print(f"Continuing to evaluate from test {evaluate_range[0]} to {evaluate_range[1] + 1}")

        display_config = DisplayConfig(show_indicator=False, print_results=False, verbose_mode=False)
        start = time()
        results = []
        for i, test_case in enumerate(test_cases, start=1):
            self.print(f"Evaluating test case {i}/{len(test_cases)}")
            result, _ = evaluate(test_cases=[test_case], metrics=self.metrics, display_config=display_config)
            results.extend(result[1])

        end = time()

        evaluation_results = self.get_evaluation_results(results, end - start)

        with open(evaluation_output_path, "w") as file:
            json.dump(evaluation_results, file, indent=4)

        self.print(f"Results saved to {evaluation_output_path}")

    def get_evaluation_results(self, results, time_taken):
        json_dict = {
            "model_for_evaluation": self.evaluation_model_name,
            "evaluation_time_seconds": time_taken
        }

        scores = defaultdict(list)
        test_reasons = defaultdict()

        # example results: ('test_results', [TestResult(name='test_case_0', success=True, metrics_data=[MetricData(name='Answer Relevancy', threshold=0.5, success=True, score=0.6, reason=None, strict_mode=False, evaluation_model='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', error=None, evaluation_cost=None, verbose_logs='Statements:\n[\n    "...",\n    "...",\n    "...",\n    "...",\n    "..."\n] \n \nVerdicts:\n[\n    {\n        "verdict": "yes",\n        "reason": null\n    },\n    {\n        "verdict": "no",\n        "reason": null\n    },\n    {\n        "verdict": "yes",\n        "reason": null\n    },\n    {\n        "verdict": "no",\n        "reason": null\n    },\n    {\n        "verdict": "yes",\n        "reason": null\n    }\n]'), MetricData(name='Faithfulness', threshold=0.5, success=True, score=0.5, reason=None, strict_mode=False, evaluation_model='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', error=None, evaluation_cost=None, verbose_logs='Truths (limit=None):\n[\n    "...",\n    "..."\n] \n \nClaims:\n[\n    "Jack Nicholson\'s \'The Dark Knight\' delves into themes of hero-visibility duality, identity manipulation, family tension, power dynamics, and social commentary. These themes collectively explore Jack\'s multifaceted journey and his place within society.\n"\n] \n \nVerdicts:\n[\n    {\n        "verdict": "yes",\n        "reason": null\n    },\n    {\n        "verdict": "no",\n        "reason": null\n    }\n]')], conversational=False, multimodal=False, input='Can you summarize the main themes of The Dark Knight for me?', actual_output='\n\n"The Dark Knight" explores several key themes:\n\n1. **Hero-Visa Duality**: Jack Nicholson, as the Dark Knight, seeks to protect others but ends up being harmed by a criminal gang that resembles the "Kingsmen of the Dead."\n\n2. **Identity and Manipulation**: The film delves into how Jack sees himself—both hero and villain—and his actions sometimes seem manipulative.\n\n3. **Family Dynamics**: There\'s a tension between Jack and Harold Hurwitz, where Jack tries to protect others but may exploit their strength.\n\n4. **Power and Control**: The Gang\'s influence is both a threat and a source of strength for Jack, highlighting power dynamics within the narrative.\n\n5. **Social Commentary**: Despite its humor, "The Dark Knight" also touches on broader social issues, offering a critical look at the corrupting effects of certain behaviors.\n\nThese themes together create a complex exploration of Jack\'s multifaceted journey and his place in a world that feels both familiar and familiarized with his past.', expected_output="The Dark Knight (2008), directed by Christopher Nolan, explores several deep and interconnected themes. The main ones include:\n\n\t1. Chaos vs. Order: The Joker represents chaos, anarchy, and unpredictability, while Batman and the authorities strive to maintain order. The film explores how fragile societal structures are when challenged by extreme forces.\n\n\t2. Moral Ambiguity and Duality: The film questions traditional notions of good and evil. Batman must bend ethical lines to fight crime, while Harvey Dent’s transformation into Two-Face shows how easily a hero can fall.\n\n\t3. Justice vs. Vengeance: Batman operates outside the law but seeks justice, whereas characters like Dent blur the line by giving in to vengeance when wronged.\n\n\t4. The Nature of Heroism: The film challenges the idea of what makes someone a hero. Batman chooses to be seen as a villain to protect Gotham's hope, highlighting the theme of self-sacrifice for the greater good.\n\n\t5. Fear and Corruption: Fear is used as both a weapon and a shield, while Gotham's institutions are portrayed as vulnerable to corruption—something both Batman and the Joker exploit in different ways.", context=None, retrieval_context=['The Dark Knight from the year 2008 (Christopher Nolan). Batman raises the stakes in his war on crime. With the help of Lt. Jim Gordon and District Attorney Harvey Dent, Batman sets out to dismantle the remaining criminal organizations that plague the streets. The partnership proves to be effective, but they soon find themselves prey to a reign of chaos unleashed by a rising criminal mastermind known to the terrified citizens of Gotham as the Joker. A gang of masked criminals rob a mafia-owned bank in Gotham City, betraying and killing each other until the sole survivor, the Joker, reveals himself as the mastermind and escapes with the money. The vigilante Batman, district attorney Harvey Dent, and police lieutenant Jim Gordon ally to eliminate Gotham\'s organized crime. Batman\'s true identity, the billionaire Bruce Wayne, publicly supports Dent as Gotham\'s legitimate protector, believing Dent\'s success will allow him to retire as Batman and romantically pursue his childhood friend Rachel Dawes—despite her being with Dent. Gotham\'s mafia bosses gather to discuss protecting their organizations from the Joker, the police, and Batman. The Joker interrupts the meeting and offers to kill Batman for half of the fortune their accountant, Lau, concealed before fleeing to Hong Kong to avoid extradition. With the help of Wayne Enterprises CEO Lucius Fox, Batman finds Lau in Hong Kong and returns him to the custody of Gotham police. His testimony enables Dent to apprehend the crime families. The bosses accept the Joker\'s offer, and he kills high-profile targets involved in the trial, including the judge and police commissioner. Although Gordon saves the mayor, the Joker threatens that his attacks will continue until Batman reveals his identity. He targets Dent at a fundraising dinner and throws Rachel out of a window, but Batman rescues her. Wayne struggles to understand the Joker\'s motives, to which his butler Alfred Pennyworth says "some men just want to watch the world burn." Dent claims he is Batman to lure out the Joker, who attacks the police convoy transporting him. Batman and Gordon apprehend the Joker, and Gordon is promoted to commissioner. At the police station, Batman interrogates the Joker, who says he finds Batman entertaining and has no intention of killing him. Having deduced Batman\'s feelings for Rachel, the Joker reveals she and Dent are being held separately in buildings rigged to explode. Batman races to rescue Rachel while Gordon and the other officers go after Dent, but they discover the Joker gave their positions in reverse. The explosives detonate, killing Rachel and severely burning Dent\'s face on one side. The Joker escapes custody, extracts the fortune\'s location from Lau, and burns it, killing Lau in the process. Coleman Reese, a consultant for Wayne Enterprises, deduces and tries to expose Batman\'s identity, but the Joker threatens to blow up a hospital unless Reese is killed. While the police evacuate hospitals and Gordon struggles to keep Reese alive, the Joker meets with a disillusioned Dent, persuading him to take the law into his own hands and avenge Rachel. Dent defers his decision-making to his now half-scarred, two-headed coin, killing the corrupt officers and the mafia involved in Rachel\'s death. As panic grips the city, the Joker reveals two evacuation ferries, one carrying civilians and the other prisoners, are rigged to explode at midnight unless one group sacrifices the other. To the Joker\'s disbelief, the passengers refuse to kill one another. Batman subdues the Joker but refuses to kill him. Before the police arrest the Joker, he says although Batman proved incorruptible, his plan to corrupt Dent has succeeded. Dent takes Gordon\'s family hostage, blaming his negligence for Rachel\'s death. He flips his coin to decide their fates, but Batman tackles him to save Gordon\'s son, and Dent falls to his death. Believing Dent is the hero the city needs and the truth of his corruption will harm Gotham, Batman takes the blame for his death and actions and persuades Gordon to conceal the truth. Pennyworth burns an undelivered letter from Rachel to Wayne that says she chose Dent, and Fox destroys the invasive surveillance network that helped Batman find the Joker. The city mourns Dent as a hero, and the police launch a manhunt for Batman.'], additional_metadata=None)])
        for test_result in results:
            # example test_result: TestResult(name='test_case_0', success=False, metrics_data=[MetricData(name='Answer Relevancy', threshold=0.5, success=True, score=1.0, reason=None, strict_mode=False, evaluation_model='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', error=None, evaluation_cost=None, verbose_logs='Statements:\n[\n    "statement 1",\n    "statement 2",\n    "statement 3",\n    "statement 4",\n    "statement 5"\n] \n \nVerdicts:\n[\n    {\n        "verdict": "yes",\n        "reason": null\n    },\n    {\n        "verdict": "yes",\n        "reason": null\n    },\n    {\n        "verdict": "yes",\n        "reason": null\n    },\n    {\n        "verdict": "yes",\n        "reason": null\n    },\n    {\n        "verdict": "yes",\n        "reason": null\n    }\n]'), MetricData(name='Faithfulness', threshold=0.5, success=False, score=0.0, reason=None, strict_mode=False, evaluation_model='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', error=None, evaluation_cost=None, verbose_logs='Truths (limit=None):\n[\n    "Truth about Dark Knight\'s appearance",\n    "Truth about Batman\'s mission",\n    "Truth about Joker\'s plan",\n    "Truth about Dark Knight\'s role",\n    "Truth about Batman\'s name",\n    "Truth about Joker\'s motivation",\n    "Truth about Dark Knight\'s goal",\n    "Truth about Batman\'s background",\n    "Truth about Joker\'s involvement",\n    "Truth about Dark Knight\'s emotional state",\n    "Truth about Batman\'s ultimate purpose"\n] \n \nClaims:\n[\n    "Jack seeks to protect others but ends up being harmed by a criminal gang resembling the Kingsmen of the Dead."\n] \n \nVerdicts:\n[\n    {\n        "verdict": "no",\n        "reason": "The actual output claims Jack seeks to protect others but ends up being harmed by a criminal gang resembling the Kingsmen of the Dead. This statement does not contradict any facts in the Retrieval Contexts because there are no related details about Jack or criminals resembling Kingsmen of the Dead."\n    }\n]')], conversational=False, multimodal=False, input='Can you summarize the main themes of The Dark Knight for me?', actual_output='\n\n"The Dark Knight" explores several key themes:\n\n1. **Hero-Visa Duality**: Jack Nicholson, as the Dark Knight, seeks to protect others but ends up being harmed by a criminal gang that resembles the "Kingsmen of the Dead."\n\n2. **Identity and Manipulation**: The film delves into how Jack sees himself—both hero and villain—and his actions sometimes seem manipulative.\n\n3. **Family Dynamics**: There\'s a tension between Jack and Harold Hurwitz, where Jack tries to protect others but may exploit their strength.\n\n4. **Power and Control**: The Gang\'s influence is both a threat and a source of strength for Jack, highlighting power dynamics within the narrative.\n\n5. **Social Commentary**: Despite its humor, "The Dark Knight" also touches on broader social issues, offering a critical look at the corrupting effects of certain behaviors.\n\nThese themes together create a complex exploration of Jack\'s multifaceted journey and his place in a world that feels both familiar and familiarized with his past.', expected_output="The Dark Knight (2008), directed by Christopher Nolan, explores several deep and interconnected themes. The main ones include:\n\n\t1. Chaos vs. Order: The Joker represents chaos, anarchy, and unpredictability, while Batman and the authorities strive to maintain order. The film explores how fragile societal structures are when challenged by extreme forces.\n\n\t2. Moral Ambiguity and Duality: The film questions traditional notions of good and evil. Batman must bend ethical lines to fight crime, while Harvey Dent’s transformation into Two-Face shows how easily a hero can fall.\n\n\t3. Justice vs. Vengeance: Batman operates outside the law but seeks justice, whereas characters like Dent blur the line by giving in to vengeance when wronged.\n\n\t4. The Nature of Heroism: The film challenges the idea of what makes someone a hero. Batman chooses to be seen as a villain to protect Gotham's hope, highlighting the theme of self-sacrifice for the greater good.\n\n\t5. Fear and Corruption: Fear is used as both a weapon and a shield, while Gotham's institutions are portrayed as vulnerable to corruption—something both Batman and the Joker exploit in different ways.", context=None, retrieval_context=['The Dark Knight from the year 2008 (Christopher Nolan). Batman raises the stakes in his war on crime. With the help of Lt. Jim Gordon and District Attorney Harvey Dent, Batman sets out to dismantle the remaining criminal organizations that plague the streets. The partnership proves to be effective, but they soon find themselves prey to a reign of chaos unleashed by a rising criminal mastermind known to the terrified citizens of Gotham as the Joker. A gang of masked criminals rob a mafia-owned bank in Gotham City, betraying and killing each other until the sole survivor, the Joker, reveals himself as the mastermind and escapes with the money. The vigilante Batman, district attorney Harvey Dent, and police lieutenant Jim Gordon ally to eliminate Gotham\'s organized crime. Batman\'s true identity, the billionaire Bruce Wayne, publicly supports Dent as Gotham\'s legitimate protector, believing Dent\'s success will allow him to retire as Batman and romantically pursue his childhood friend Rachel Dawes—despite her being with Dent. Gotham\'s mafia bosses gather to discuss protecting their organizations from the Joker, the police, and Batman. The Joker interrupts the meeting and offers to kill Batman for half of the fortune their accountant, Lau, concealed before fleeing to Hong Kong to avoid extradition. With the help of Wayne Enterprises CEO Lucius Fox, Batman finds Lau in Hong Kong and returns him to the custody of Gotham police. His testimony enables Dent to apprehend the crime families. The bosses accept the Joker\'s offer, and he kills high-profile targets involved in the trial, including the judge and police commissioner. Although Gordon saves the mayor, the Joker threatens that his attacks will continue until Batman reveals his identity. He targets Dent at a fundraising dinner and throws Rachel out of a window, but Batman rescues her. Wayne struggles to understand the Joker\'s motives, to which his butler Alfred Pennyworth says "some men just want to watch the world burn." Dent claims he is Batman to lure out the Joker, who attacks the police convoy transporting him. Batman and Gordon apprehend the Joker, and Gordon is promoted to commissioner. At the police station, Batman interrogates the Joker, who says he finds Batman entertaining and has no intention of killing him. Having deduced Batman\'s feelings for Rachel, the Joker reveals she and Dent are being held separately in buildings rigged to explode. Batman races to rescue Rachel while Gordon and the other officers go after Dent, but they discover the Joker gave their positions in reverse. The explosives detonate, killing Rachel and severely burning Dent\'s face on one side. The Joker escapes custody, extracts the fortune\'s location from Lau, and burns it, killing Lau in the process. Coleman Reese, a consultant for Wayne Enterprises, deduces and tries to expose Batman\'s identity, but the Joker threatens to blow up a hospital unless Reese is killed. While the police evacuate hospitals and Gordon struggles to keep Reese alive, the Joker meets with a disillusioned Dent, persuading him to take the law into his own hands and avenge Rachel. Dent defers his decision-making to his now half-scarred, two-headed coin, killing the corrupt officers and the mafia involved in Rachel\'s death. As panic grips the city, the Joker reveals two evacuation ferries, one carrying civilians and the other prisoners, are rigged to explode at midnight unless one group sacrifices the other. To the Joker\'s disbelief, the passengers refuse to kill one another. Batman subdues the Joker but refuses to kill him. Before the police arrest the Joker, he says although Batman proved incorruptible, his plan to corrupt Dent has succeeded. Dent takes Gordon\'s family hostage, blaming his negligence for Rachel\'s death. He flips his coin to decide their fates, but Batman tackles him to save Gordon\'s son, and Dent falls to his death. Believing Dent is the hero the city needs and the truth of his corruption will harm Gotham, Batman takes the blame for his death and actions and persuades Gordon to conceal the truth. Pennyworth burns an undelivered letter from Rachel to Wayne that says she chose Dent, and Fox destroys the invasive surveillance network that helped Batman find the Joker. The city mourns Dent as a hero, and the police launch a manhunt for Batman.'], additional_metadata=None)
            reasons = defaultdict(list)
            for metric_data in test_result.metrics_data:
                # example metric_data: name='Answer Relevancy' threshold=0.5 success=True score=1.0 reason=None strict_mode=False evaluation_model='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' error=None evaluation_cost=None verbose_logs='Statements:\n[\n    "...",\n    "..."\n] \n \nVerdicts:\n[\n    {\n        "verdict": "yes",\n        "reason": "This statement is unrelated to summarizing themes. Themes are subjective and depend on interpretation."\n    }\n]'
                metric_name = metric_data.name
                metric_score = metric_data.score
                metric_success = metric_data.success
                metric_threshold = metric_data.threshold

                metric_reason = metric_data.reason

                scores[metric_name].append(metric_score)
                reasons[metric_name].append(metric_reason)

                metric_error = metric_data.error
                if metric_error is not None:
                    warnings.warn(f"Error in metric {metric_name}: {metric_error}")
            test_reasons[test_result.name] = reasons

        self.print(f"Evaluation time: {time_taken}s")
        self.print("Final results:")
        for metric_name, metric_scores in scores.items():
            score_average = np.mean(scores[metric_name])
            score_median = np.median(scores[metric_name])

            score_min = np.min(scores[metric_name])
            score_max = np.max(scores[metric_name])

            score_deviation = np.std(scores[metric_name])

            self.print(f"----------------------------------")
            self.print(f"Metric: {metric_name}")
            self.print(f"\tAverage: {score_average}")
            self.print(f"\tMedian: {score_median}")
            self.print(f"\tMaximum: {score_max}")
            self.print(f"\tMinimum: {score_min}")
            self.print(f"\tStd. Deviation: {score_deviation}")

            if metric_name not in json_dict:
                json_dict[metric_name] = dict()
            json_dict[metric_name]["average"] = score_average
            json_dict[metric_name]["median"] = score_median
            json_dict[metric_name]["minimum"] = score_min
            json_dict[metric_name]["maximum"] = score_max
            json_dict[metric_name]["standard_deviation"] = score_deviation
        json_dict["reasons"] = test_reasons

        return json_dict


if __name__ == "__main__":
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-32B

    # meta-llama/Llama-3.1-8B-Instruct
    # meta-llama/Llama-3.1-70B-Instruct

    # meta-llama/Llama-3.3-70B-Instruct

    # 01-ai/Yi-34B-Chat
    # Qwen/Qwen3-14B

    # Arguments:
    # 1. model to evaluate (without json)
    # 2. model used to evaluate (from huggingface)
    # 3. [OPTIONAL] evaluate start test
    # 4. [OPTIONAL] evaluate end test (inclusive [start, end])

    # model_to_evaluate = "qwen_naive"
    # model_name = "Qwen/Qwen3-14B"

    results_path = os.path.join("..", "final_results_for_evaluation", f"{model_to_evaluate}.json")
    evaluation_results_path = os.path.join("..", "final_results_for_evaluation",
                                           f"{model_to_evaluate}_evaluation{evaluate_range}.json")
    verbose_mode = True

    # model_name = None
    include_reason = True

    evaluation = EvaluationWithLLM(model_name_for_evaluation=model_name, include_reason=include_reason,
                                   verbose_mode=verbose_mode, hugging_face_token=hf_key)
    evaluation.evaluate_results(results_path, evaluation_results_path, evaluate_range=evaluate_range)