from typing import Any, Dict
import requests
import transformers
import torch


class Rag:
    def __init__(self):
        self.headers = {
            "accept": "application/json",
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJjNzA3OTdmMTNkOGEyYjE2ODZhM2MxZTI0MzBmYmI1NCIsIm5iZiI6MTc0NDk4ODE2NC4yMjU5OTk4LCJzdWIiOiI2ODAyNjgwNDJjODVlNzk2NjM5OWJkYTYiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.prH_dcerfMwd_oxlbU6qBIaH5tqkBO3yu-z09XirBAE"
        }
        self.tmdb_movie_url = "https://api.themoviedb.org/3/search/movie?query={movie_title}&include_adult=false&language=en-US&page=1"
        self.tmdb_person_url = "https://api.themoviedb.org/3/search/person?query={person_name}&include_adult=false&language=en-US&page=1"
        
        self.cache = {}

    def retrieve_tmdb_movie_data(self, title: str) -> Any:
        url = self.tmdb_movie_url.format(movie_title=title)
        cache = self.cache.get(url, None)
        if cache:
            return cache
        
        response = requests.get(url, headers=self.headers)
        json = response.json()
        documents = json.get("results", None)
        self.cache[url] = documents
        return documents
    
    def retrieve_tmdb_person_data(self, person: str) -> Any:
        url = self.tmdb_person_url.format(person_name=person)
        cache = self.cache.get(url, None)
        if cache:
            return cache
        
        response = requests.get(url, headers=self.headers)
        json = response.json()
        documents = json.get("results", None)
        self.cache[url] = documents
        return documents
    
    def get_movie_release_date(self, title: str) -> Dict[str, str]:
        """
        Get the release date for a movie.

        Args:
            title: The title of the movie.

        Returns:
            The release dates of all movies with that or similar title, in format 'YYYY-mm-dd'.
        """
        docs = self.retrieve_tmdb_movie_data(title)
        results = {}
        for movie in docs:
            title = movie.get("title", None)
            if title is None:
                continue

            release_date = movie.get("release_date", "unknown")
            results[title] = release_date
        return results

class HermesBot:
    def __init__(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "NousResearch/Hermes-3-Llama-3.2-3B",
            trust_remote_code=True
        )
        quantization_config = transformers.BitsAndBytesConfig(load_in_4bit=True)
        self.model = transformers.LlamaForCausalLM.from_pretrained(
            "NousResearch/Hermes-3-Llama-3.2-3B",
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            attn_implementation="flash_attention_2"
        )
        self.rag = Rag()
        self.tools = [
            self.rag.get_movie_release_date
        ]
    
    def generate_response(self, user_prompt: str):
        messages = [
            {"role": "system", "content": "You are a chatbot that helps the user with anything related to movies. You should decide what kind of information to retrieve in order to help the user."},
            {"role": "user", "content": user_prompt}
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tools=self.tools,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = {key: value for key,value in inputs.items()}
        tool_outputs = self.model.generate(**inputs, max_new_tokens=512)
        tool_response = self.tokenizer.decode(tool_outputs[0][len(inputs["input_ids"][0]):])
        print(tool_response)


if __name__ == "__main__":
    hermes = HermesBot()
    prompt = "When did the movie Inception release?"
    hermes.generate_response(prompt)