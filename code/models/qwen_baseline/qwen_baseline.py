import threading
from typing import Dict, Iterator, Tuple
from ..model import Model
import transformers

class QwenBaseline(Model):
    def __init__(self, name, folder, datafolder, outname):
        self.name = name
        self.folder = folder
        self.datafolder = datafolder
        self.model_label = name 
        self.outname = outname
        self.chat_history = []
        self.context = None
        self.mode = "baseline"
        self.sources = "/"
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_label)
        self.temperature = 0.7
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_label,
            torch_dtype="auto",
            device_map="auto"
        )
        self.pad_token_id = self.tokenizer.eos_token_id

        self.generation_thread = None

        with open("./models/qwen_baseline/prompt_template_qwen.txt", "r") as fd:
            self.prompt_template = fd.read()

    def train(self):
        pass

    def reply(self, prompt) -> Tuple[str, Dict]:
        return self.prompt_nonstream(prompt)

    def prompt_stream(self, prompt: str, data: str = "") -> Tuple[transformers.TextIteratorStreamer, Dict]:
        return self.prompt_nonstream(prompt, data)
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
        final_prompt = self.prompt_template.format(data=data, query=prompt)
        final_prompt = [
            {
                "role": "system", "content": "You are an AI assistant tasked with helping the user on film or series-related questions. Read the following data and answer the question. If you cannot infer information from the data, do not answer the question.",
            },
            {
                "role": "user", "content": prompt
            }
        ]
        text = self.tokenizer.apply_chat_template(
            final_prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        input_tokens = self.tokenizer(text, return_tensors="pt").to('cuda')

        outputs = self.model.generate(
            **input_tokens,
            max_new_tokens=32768,
            pad_token_id=self.pad_token_id,
            temperature=self.temperature,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.2,
            length_penalty=0.9
        )

        final_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        return final_output, {"context":""}
