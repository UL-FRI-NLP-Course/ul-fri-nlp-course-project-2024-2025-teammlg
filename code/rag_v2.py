import json
import re
from typing import Any, List
import transformers
import scraper_advanced


class RagV2:
    def __init__(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-8B",
            trust_remote_code=True
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-8B",
            torch_dtype="auto",
            device_map="auto"
        )
        self.tools = [
            scraper_advanced.get_movie_release_date,
            scraper_advanced.get_movie_genres,
            scraper_advanced.get_movie_cast_list,
            scraper_advanced.get_movie_reviews,
            scraper_advanced.get_movie_streaming_services,
            scraper_advanced.get_movie_summary,
            scraper_advanced.get_person_credits,
            scraper_advanced.get_similar_movies
        ]

    @staticmethod
    def _handle_tool_calls(tool_response: str) -> List[Any]:
        tool_calls = re.findall(r"<tool_call>(?P<tool_call>[\S\s]*?)</tool_call>", tool_response)
        retrieved_data = []
        for tool_call in tool_calls:
            try:
                tool_call_json = json.loads(tool_call)
                name = tool_call_json.get("name", None)
                if name is None:
                    return ""
                arguments = tool_call_json.get("arguments", {})
                f = getattr(scraper_advanced, name)
                data = f(**arguments)
                retrieved_data.append(data)
            except Exception as e:
                print(e)
        return retrieved_data

    def get_context(self, user_prompt: str) -> List[Any]:
        messages = [
            {"role": "system", "content": "You are a function calling AI chatbot. You assist the user with anything related to movies."},
            {"role": "user", "content": user_prompt}
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tools=self.tools,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to('cuda')
        tool_outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.8,
            repetition_penalty=1.1,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id
        )
        tool_response = self.tokenizer.decode(
            tool_outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True,
            clean_up_tokenization_space=True
        )
        data = self._handle_tool_calls(tool_response)
        return data


if __name__ == "__main__":
    hermes = RagV2()
    prompt = "When did the movie Challengers release and what are its genres?"
    hermes.get_context(prompt)