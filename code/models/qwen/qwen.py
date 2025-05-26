from typing import Tuple
from ..model import Model
from scraper import *
from POStagger import *
from summarizer import *
from rag import *
from memory import *
import transformers

class QwenChatBot(Model):
    def __init__(self, name, folder, datafolder, outname, sources=["tmdb", "letterboxd", "justwatch", "wiki"], mode="naive"):
        self.name = name
        self.folder = folder
        self.datafolder = datafolder
        self.sources = sources
        self.model_label = name
        self.outname = outname
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

        if mode == "advanced":
            self.rag = RagV2(self.tokenizer, self.model)
        else:
            self.rag = Rag(self.mode, self.datafolder, self.outname, self.sources)
        with open("./models/qwen/prompt_template_qwen.txt", "r") as fd:
            self.prompt_template = fd.read()

    def train(self):
        pass

    def reply(self, prompt):
        return self.prompt_nonstream(prompt)

    def prompt_stream(self, prompt: str, data: str = "") -> Tuple[str, Dict]:
        return self.prompt_nonstream(prompt, data)

    def prompt_nonstream(self, prompt: str, data: str = "") -> Tuple[str, Dict]:
        self.context, state = self.rag.get_context(prompt)

        enhanced_prompt = f"{prompt}\n<data>{self.context}</data>"
        system_prompt = "You are an AI chatbot, assisting user with anything related to movies. You may only use information provided to you inside the <data> tags."

        messages = [
            {
                "role": "system", "content": system_prompt,
            },
            {
                "role": "user", "content": enhanced_prompt
            }
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
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

        self.session.add_user_query(prompt)
        self.session.add_assistant_response(str(final_output))

        return final_output, state
