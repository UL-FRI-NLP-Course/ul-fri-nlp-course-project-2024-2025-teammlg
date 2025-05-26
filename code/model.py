from enum import Enum
import logging
import os
import time
from typing import List, TypedDict

import transformers


class Role(Enum):
    User = "user"
    System = "system"
    Assistant = "assistant"


class ChatElement(TypedDict):
    role: Role
    content: str


class Model:
    """A base class that handles LLM initialization
    and holds text generation and chat methods"""

    def __init__(
        self,
        label: str,
        outputs_directory: str = ".",
        temperature: float = 0.6,
        max_new_tokens: int = 1024,
        logging_level: int = logging.INFO,
        log_to_console: bool = False
    ):
        # Names and paths
        name = label.split("/")[-1]
        name = name.replace(".", "")
        self.label = label
        self.name = name
        self.outputs_directory = outputs_directory

        # Logging
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging_level)
        logging_output_path = os.path.join(self.outputs_directory, f"{self.name}.log")
        self.logger.addHandler(logging.FileHandler(logging_output_path))
        if log_to_console:
            self.logger.addHandler(logging.StreamHandler())

        # Model initialization
        self.logger.debug("Initializing tokenizer...")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.label,
            padding_side="left"
        )
        self.logger.debug("Initializing model...")
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.label,
            device_map="auto",
            torch_dtype="auto"
        )
        # TODO: Try model quantization! (https://huggingface.co/docs/transformers/llm_tutorial#default-generate)

        # Generation parameters
        self.temperature = temperature
        self.pad_token_id = self.tokenizer.eos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.max_new_tokens = max_new_tokens

        self.logger.info("Initialization complete")
    
    def __str__(self) -> str:
        return f"Model {self.label}"

    def set_chat_template(self, template: str):
        """Allows for changing the default chat template to a custom one"""
        self.tokenizer.chat_template = template
        self.logger.info("Set new chat template")
    
    def chat(self, chat: List[ChatElement], generation_prompt: bool = True, sample: bool = False) -> str:
        """This returns a string response of the LLM on the provided chat query"""
        self.logger.debug("Tokenizing input...")
        input_tokens = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=generation_prompt,
            tokenize=False
        )
        input_tokens = self.tokenizer(input_tokens, return_tensors="pt").to("cuda")

        self.logger.debug("Generating response...")
        start_time = time.perf_counter()
        outputs = self.model.generate(
            **input_tokens,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.pad_token_id,
            temperature=self.temperature,
            eos_token_id=self.eos_token_id,
            do_sample=sample
        )
        end_time = time.perf_counter()
        self.logger.debug(f"Generation complete in {end_time - start_time} s")

        self.logger.debug("Reconstructing output from tokens...")
        final_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        self.logger.debug("Response complete!")
        return final_output
    
    def query(self, query: str) -> str:
        """This differs from the `chat` method in that the query can be an arbitrary
        string. Query should be formatted correctly beforehand. This covers all non-chat
        queries and custom-formatted prompts"""
        self.logger.debug("Tokenizing input...")
        input_tokens = self.tokenizer([query], padding=True, return_tensors="pt").to("cuda")
        
        self.logger.debug("Generating response...")
        start_time = time.perf_counter()
        outputs = self.model.generate(
            **input_tokens,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.pad_token_id,
            temperature=self.temperature,
            eos_token_id=self.eos_token_id
        )
        end_time = time.perf_counter()
        self.logger.debug(f"Generation complete in {end_time - start_time} s")

        self.logger.debug("Reconstructing output from tokens...")
        final_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        self.logger.debug("Response complete!")
        return final_output