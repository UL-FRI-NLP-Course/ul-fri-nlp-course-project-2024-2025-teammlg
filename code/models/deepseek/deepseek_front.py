from typing import Iterator, Tuple
from ..model import Model
from scraper import *
from POStagger import *
from summarizer import *
from rag import *
from memory import *
import transformers

class DeepSeekFilmChatBot(Model):
    def __init__(self, name, folder, datafolder, outname, sources=["tmdb", "letterboxd", "justwatch", "wiki"], mode="naive"):
        print("Model init start")
        self.name = name
        self.folder = folder
        self.datafolder = datafolder
        self.sources = sources
        self.model_label = name  # The name of the model for Ollama to download (all models here: https://ollama.com/search)
        self.outname = outname
        self.chat_history = []
        self.mode = mode
        self.context = None
        self.session = Memory()
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_label)
        self.temperature = 0.6
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_label,
            device_map="auto",
            torch_dtype="auto"
        )
        self.pad_token_id = self.tokenizer.eos_token_id
        
        self.generation_thread = None 

        with open("./models/deepseek/prompt_template_deepseek.txt", "r") as fd:
            self.prompt_template = fd.read()
        print("Model init done")

    def train(self):
        pass

    def reply(self, prompt):
        return self.prompt_nonstream(prompt)

    def prompt_stream(self, prompt: str, data: str = "") -> Tuple[str, Dict]:
        rag = Rag(prompt, self.mode, self.datafolder, self.outname, self.sources)
        self.context, state = rag.get_context()

        chat = self.session.get_chat_history()
        chat.append({
            "role": "system",
            "content": f"Here is the available data:\n\n{data}\n\nGiven the available data and no other information, answer the user query."
        })
        chat.append({
            "role": "user",
            "content": prompt
        })

        input_tokens = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to('cuda')
        
        outputs = self.model.generate(
            **input_tokens,
            max_new_tokens=512,
            pad_token_id=self.pad_token_id,
            temperature=self.temperature
        )

        final_output = ""
        for i in range(len(outputs)):
            output_text = self.tokenizer.decode(outputs[i])
            final_output += output_text
        
        split_on_think_end = final_output.split("</think>")
        if len(split_on_think_end) > 1:
            final_output = split_on_think_end[-1]

        # here we have an option not to remember a potentially bad answer (if we come up with a suitable metric)
        self.session.add_user_query(prompt)
        self.session.add_assistant_response(str(final_output))
    
        return final_output, state

    def prompt_nonstream(self, prompt: str, data: str = "") -> Tuple[str, Dict]:
        return self.prompt_nonstream(prompt, data)
        rag = Rag(prompt, self.mode, self.datafolder, self.outname, self.sources)
        self.context, state = rag.get_context()

        print("Tokenizer start")
        messages = [{"role": "user", "content": prompt}]

        input_tokens = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        

        input_tokens = self.tokenizer(input_tokens, return_tensors="pt").to("cuda")
        print("Tokenizer done")
        
        outputs = self.model.generate(
            **input_tokens,
            max_new_tokens=32768,
            pad_token_id=self.pad_token_id,
            temperature=self.temperature,
            eos_token_id=self.tokenizer.eos_token_id
        )
        

        final_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        print("Generation done")
        
        self.session.add_user_query(prompt)
        self.session.add_assistant_response(str(final_output))

        return final_output, state