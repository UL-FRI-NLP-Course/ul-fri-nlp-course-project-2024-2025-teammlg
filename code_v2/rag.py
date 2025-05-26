import json
import re
import sys
from typing import Any, Callable, List
import scraper
import spacy


class Rag:
    def __init__(self, output_directory: str):
        self.output_directory = output_directory
        self.tools = [
            scraper.get_movie_release_date,
            scraper.get_movie_genres,
            scraper.get_movie_cast_list,
            scraper.get_movie_reviews,
            scraper.get_movie_streaming_services,
            scraper.get_movie_summary,
            scraper.get_person_credits,
            scraper.get_similar_movies
        ]
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            print(e, file=sys.stderr)
            self.nlp = None
    
    def get_retrieval_tools(self) -> List[Callable]:
        return self.tools
    
    @staticmethod
    def _handle_tool_calls(tool_response: str) -> List[Any]:
        """Parses the instructions JSON, executes all calls with appropriate arguments
        and returns the results in a list.

        Args:
            tool_response (str): An instructions string (basically, what LLM output)

        Returns:
            List[Any]: A list of data, each element in list corresponds to one
            function call that the LLM requested
        """
        tool_calls = re.findall(r"<tool_call>(?P<tool_call>[\S\s]*?)</tool_call>", tool_response)
        retrieved_data = []
        for tool_call in tool_calls:
            try:
                tool_call_json = json.loads(tool_call)
                name = tool_call_json.get("name", None)
                if name is None:
                    return ""
                arguments = tool_call_json.get("arguments", {})
                f = getattr(scraper, name)
                data = f(**arguments)
                retrieved_data.append(data)
            except Exception as e:
                print(e, file=sys.stderr)
        return retrieved_data
    
    def get_context_from_tools(self, instructions: str) -> List[Any]:
        """Gets the instructions about which functions to call and executes those calls

        Args:
            instructions (str): A string of JSON-formatted function calls

        Returns:
            Any: A list of returned data (should be convertable to string)
        """
        data = self._handle_tool_calls(instructions)
        return data
    
    def get_simple_context(self, prompt: str) -> List[Any]:
        entities = []

        if self.nlp:
            document = self.nlp(prompt)
            entities = [entity.text for entity in document.ents]

        documents = []
        for entity in entities:
            tmdb = ""
            letterboxd = []
            wikipedia = ""
            justwatch = ""
            documents.append(tmdb)
            documents.append(letterboxd)
            documents.append(wikipedia)
            documents.append(justwatch)
        
        return documents
    
    def data_to_str(self, data: Any) -> str:
        """Converts data to string format, suitable for LLM.

        Args:
            data (Any): Data, usually from self.get_context(...)

        Returns:
            str: A string representing the input data
        """
        return str(data)