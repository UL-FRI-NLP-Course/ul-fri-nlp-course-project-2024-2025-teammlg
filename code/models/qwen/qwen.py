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

        self.rag = Rag(self.mode, self.datafolder, self.outname, self.sources)

        with open("./models/qwen/prompt_template_qwen.txt", "r") as fd:
            self.prompt_template = fd.read()

    def train(self):
        pass

    def reply(self, prompt):
        return self.prompt_nonstream(prompt)

    def prompt_stream(self, prompt: str, data: str = "") -> Tuple[str, Dict]:
        return self.prompt_nonstream(prompt, data)
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
        
        text = self.tokenizer.apply_chat_template(
            chat,
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
        self.session.add_user_query(prompt)
        self.session.add_assistant_response(str(final_output))
    
        return final_output, state

    def prompt_nonstream(self, prompt: str, data: str = "") -> Tuple[str, Dict]:
        self.context, state = self.rag.get_context(prompt)

        messages = [
            {
                "role": "system", "content": "You are an AI assistant tasked with helping the user on film or series-related questions. Read the following data and answer the question. If you cannot infer information from the data, do not answer the question.",
            },
            {
                "role": "user", "content": prompt
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

        """pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            use_cache=False,
            truncation=True,
            device_map="auto",
            max_new_tokens=32768,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=0.6
        )

        with open("./models/qwen/prompt_template_qwen.txt", "r") as fd:
            template = fd.read()

        template = template.format(data=self.context, query=prompt)

        output_dict = pipeline(template)
        final_output = output_dict[0]["generated_text"][len(template) :]

        return final_output, state"""
