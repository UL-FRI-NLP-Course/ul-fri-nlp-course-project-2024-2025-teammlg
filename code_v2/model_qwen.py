from typing import Any, Callable, List

import transformers
from model import Model, ModelAnswer


class QwenModel(Model):
    def __init__(
            self,
            output_directory: str,
            uses_memory: bool = False,
            memory_capacity: int = 5,
            model_label: str = "Qwen/Qwen3-8B"
        ):
        super().__init__(output_directory, uses_memory, memory_capacity)
        self.model_label = model_label
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_label)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_label,
            torch_dtype="auto",
            device_map="auto"
        )
        self.answer_temperature = 0.7
        self.tool_temperature= 0.8

        self.answer_tokens = 32768
        self.tool_tokens = 512

        self.logger.info("Model set to Qwen")
    
    def answer_prompt(self, prompt: str, data: Any = None, baseline: bool = True) -> ModelAnswer:
        self.logger.info("=========== PROMPT ANSWERING ============")

        system_prompt = "You are an AI chatbot, assisting user with anything related to movies."
        if baseline:
            system_prompt += " You may only use information provided to you inside the <data> tags."
        
        self.logger.info(f"System prompt: {system_prompt}")

        final_prompt = prompt
        if data:
            final_prompt = f"{prompt}\n<data>{data}</data>"
        
        self.logger.info(f"Final user prompt: {final_prompt}")
        
        chat = self.form_chat(system_prompt, final_prompt)

        self.logger.debug(f"Chat: {chat}")
        self.logger.info("Tokenizing...")

        text = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        input_tokens = self.tokenizer(text, return_tensors="pt").to("cuda")

        self.logger.info("Generating...")
        
        outputs = self.model.generate(
            **input_tokens,
            max_new_tokens=32768,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=self.answer_temperature,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.2,
            # length_penalty=0.9
        )
        
        self.logger.info("Decoding...")

        final_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        self.logger.info(f"Generated response: {final_output}")

        if self.uses_memory:
            self.save_to_memory(prompt, "user")
            self.save_to_memory(final_output, "assistant")
        
        return {
            "assistant_response": final_output,
            "final_prompt": final_prompt,
            "system_prompt": system_prompt
        }
    
    def answer_function_calling(self, prompt: str, tools: List[Callable]) -> str:
        self.logger.info("=========== FUNCTION CALLING ============")

        system_prompt = "You are a function calling AI chatbot. You assist the user with anything related to movies."
        chat = self.form_chat(system_prompt, prompt)

        self.logger.debug(f"Chat: {chat}")
        self.logger.info("Tokenizing...")

        inputs = self.tokenizer.apply_chat_template(
            chat,
            tools=tools,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to('cuda')

        self.logger.info("Generating...")
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=self.tool_temperature,
            repetition_penalty=1.1,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id
        )

        self.logger.info("Decoding...")

        response = self.tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True,
            clean_up_tokenization_space=True
        )

        self.logger.info(f"Response: {response}")
        return response