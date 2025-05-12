import threading
from typing import Dict, Tuple
from ..model import Model
import transformers

class DeepSeekBaseline(Model):
    def __init__(self, name, folder, datafolder):
        self.name = name
        self.folder = folder
        self.datafolder = datafolder
        self.model_label = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # The name of the model for Ollama to download (all models here: https://ollama.com/search)
        self.chat_history = []
        self.outname = "deepseek1_5_baseline"
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

    def prompt_stream(self, prompt: str, data: str = "") -> Tuple[transformers.TextIteratorStreamer, str]:
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
        final_prompt = self.prompt_template.format(data=data, query=prompt)

        input_tokens = self.tokenizer.apply_chat_template(
            final_prompt,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to('cuda')
        
        outputs = self.model.generate(
            **input_tokens,
            max_new_tokens=512,
            pad_token_id=self.pad_token_id
        )

        final_output = ""
        for i in range(len(outputs)):
            output_text = self.tokenizer.decode(outputs[i])
            final_output += output_text
        
        return final_output, {"context":""}