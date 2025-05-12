from typing import Iterator, Tuple
from ..model import Model
from scraper import *
from POStagger import *
from summarizer import *
from rag import *
from memory import *
import transformers

class QwenChatBot(Model):
    def __init__(self, name, folder, datafolder, sources=["tmdb", "letterboxd", "justwatch"], mode="naive"):
        self.name = name
        self.folder = folder
        self.datafolder = datafolder
        self.sources = sources
        self.model_label = "Qwen/Qwen3-8B"
        self.outname = "qwen1_8"
        self.chat_history = []
        self.mode = mode
        self.context = None
        self.session = Memory(initial_template="You are an AI assistant tasked with helping the user on film or series-related questions. Read the following data and conversation history and answer the question. \n\nData: {data}\n\n{history}\nAssistant:")
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_label)
        self.temperature = 0.7
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_label,
            torch_dtype="auto",
            device_map="auto"
        )
        self.pad_token_id = self.tokenizer.eos_token_id

        with open("./models/qwen/prompt_template_qwen.txt", "r") as fd:
            self.prompt_template = fd.read()

    def train(self):
        pass

    def reply(self, prompt):
        return self.prompt_nonstream(prompt)

    def prompt_stream(self, prompt: str, data: str = "") -> Tuple[str, Dict]:
        rag = Rag(prompt, self.mode, self.datafolder, self.outname, self.sources)
        self.context, state = rag.get_context()

        final_prompt = self.session.get_template(self.context, prompt)
        
        text = self.tokenizer.apply_chat_template(
            final_prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        input_tokens = self.tokenizer([text], return_tensors="pt").to('cuda')

        outputs = self.model.generate(
            **input_tokens,
            max_new_tokens=32768,
            pad_token_id=self.pad_token_id,
            temperature=self.temperature
        )

        final_output = ""
        for i in range(len(outputs)):
            output_text = self.tokenizer.decode(outputs[i])
            final_output += output_text

        # here we have an option not to remember a potentially bad answer (if we come up with a suitable metric)
        self.session.add(prompt, str(final_output))
    
        return final_output, state

    def prompt_nonstream(self, prompt: str, data: str = "") -> Tuple[str, Dict]:
        rag = Rag(prompt, self.mode, self.datafolder, self.outname, self.sources)
        self.context, state = rag.get_context()

        final_prompt = self.prompt_template.format(data=self.context, query=prompt)

        text = self.tokenizer.apply_chat_template(
            final_prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        input_tokens = self.tokenizer([text], return_tensors="pt").to('cuda')

        outputs = self.model.generate(
            **input_tokens,
            max_new_tokens=32768,
            pad_token_id=self.pad_token_id,
            temperature=self.temperature
        )

        final_output = ""
        for i in range(len(outputs)):
            output_text = self.tokenizer.decode(outputs[i])
            final_output += output_text

        return final_output, state
