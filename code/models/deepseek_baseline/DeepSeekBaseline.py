import threading
from typing import Dict, Tuple
import transformers
import accelerate

class DeepSeekBaseline():
    def __init__(self, name, folder, datafolder, outname):
        self.name = name
        self.folder = folder # deepseek-ai/DeepSeek-R1-Distill-Llama-8B
        self.datafolder = datafolder # deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
        self.model_label = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # The name of the model for Ollama to download (all models here: https://ollama.com/search)
        self.chat_history = []
        self.outname = outname
        self.context = None
        self.mode = "baseline"
        self.sources = "/"

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_label)
        self.temperature = 0.6
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_label,
            device_map="auto",
            torch_dtype="auto"
        )
        self.pad_token_id = self.tokenizer.eos_token_id

        self.generation_thread = None

        with open("./models/deepseek_baseline/prompt_template_deepseek.txt", "r") as fd:
            self.prompt_template = fd.read()

    def train(self):
        pass

    def reply(self, prompt):
        return self.prompt_nonstream(prompt)

    def prompt_stream(self, prompt: str, data: str = "") -> Tuple[transformers.TextIteratorStreamer, Dict]:
        """Feeds the prompt to the model, returning its response as a stream iterator"""
        final_prompt = self.prompt_template.format(data=data, query=prompt)

        inputs = self.tokenizer.apply_chat_template(
            final_prompt,
            add_generation_prompt=True,
            tokenize=False
        )

        streamer = transformers.TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        input_tokens = self.tokenizer([inputs], return_tensors="pt").to('cuda')

        generation_arguments = {
            'max_new_tokens': 512,
            'streamer': streamer,
            'temperature': self.temperature,
            **input_tokens
        }

        self.generation_thread = threading.Thread(
            target=self.model.generate,
            kwargs=generation_arguments
        )

        self.generation_thread.start()

        return streamer, {"context":""}
    
    def join_thread(self):
        if self.generation_thread is not None and self.generation_thread.is_alive():
            self.generation_thread.join()

    def prompt_nonstream(self, prompt: str, data: str = "") -> Tuple[str, Dict]:
        """Feeds the prompt to the model, returning its response"""
        #final_prompt = self.prompt_template.format(data=data, query=prompt)

        messages = [{"role": "user", "content": prompt}]

        # Apply chat template to get the string-formatted prompt
        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        # Tokenize to get input_ids and attention_mask
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=64,
            pad_token_id=self.pad_token_id,
            temperature=self.temperature
        )

        """final_output = ""
        for i in range(len(outputs)):
            output_text = self.tokenizer.decode(outputs[i])
            final_output += output_text"""

        final_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return final_output, {"context":""}

if __name__ == "__main__":
    llm = DeepSeekBaseline(None, None, None, None)
    print(llm.reply("What color is a fire truck?"))