from typing import List
import vllm
import torch


class TestModel:
    def __init__(self):
        self.llm = vllm.LLM(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            dtype=torch.bfloat16,
            trust_remote_code=True,
            quantization="bitsandbytes"
        )
        self.sampling_parameters = vllm.SamplingParams(temperature=0.8, top_p=0.95)
    
    def reply(self, prompts: List[str]) -> str:
        outputs = self.llm.generate([prompts], self.sampling_parameters)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}\nGenerated: {generated_text!r}\n")


if __name__ == "__main__":
    prompts = [
        "Hi! How are you?",
        "Who are you?",
        "Do you like movies?"
    ]
    model = TestModel()
    model.reply(prompts)