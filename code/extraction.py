from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Optional, TypedDict

import spacy


class ExtractedData(TypedDict):
    movies: List[str]
    people: List[str]
    other: List[str]


class Extractor(ABC):
    @abstractmethod
    def extract_useful_information(self, user_prompt: str) -> ExtractedData:
        ...


class NullExtractor(Extractor):
    def __init__(
        self,
        label: str = "EntityExtractor",
        logging_level: int = logging.INFO
    ):
        super().__init__()
        self.logger = logging.getLogger(label)
        self.logger.setLevel(logging_level)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.info("Initialized null extractor")
    
    def extract_useful_information(self, user_prompt: str) -> ExtractedData:
        return {
            "movies": [],
            "people": [],
            "other": []
        }


class EntityExtractor(Extractor):
    def __init__(
        self,
        label: str = "EntityExtractor",
        accurate: bool = False,
        logging_level: int = logging.INFO
    ):
        super().__init__()
        self.logger = logging.getLogger(label)
        self.logger.setLevel(logging_level)
        self.logger.addHandler(logging.StreamHandler())

        if not accurate:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.debug("Loaded spacy model (en_core_web_sm)")
            except Exception as e:
                self.logger.error(f"Could not load spacy model ('en_core_web_sm'): {e}")
                self.nlp = None
        else:
            try:
                self.nlp = spacy.load("en_core_web_trf")
                self.logger.debug("Loaded spacy model ('en_core_web_trf')")
            except Exception as e:
                self.logger.error(f"Could not load spacy model ('en_core_web_trf'): {e}")
                self.nlp = None
        
        self.logger.debug("Retrieving stopwords...")
        with open('./data/stopwords-en.txt', "r") as f:
            self.stop_words = f.readlines()
            self.stop_words = [word.strip() for word in self.stop_words]
        self.logger.info("Initialized entity extractor")
    
    def extract_useful_information(self, user_prompt: str) -> ExtractedData:
        extracted_data = {
            "movies": [],
            "people": [],
            "other": []
        }

        if self.nlp:
            document = self.nlp(user_prompt)
            for entity in document.ents:
                extracted_data["people"].append(entity.text)
                extracted_data["movies"].append(entity.text)
                self.logger.debug(f"Extracted entity {entity.text} (type {entity.label_})")
            
            tokens = [word.lemma_ for word in document if word.lemma_ not in self.stop_words]
            extracted_data["other"] = list(set(tokens))
        else:
            self.logger.warning("Spacy pipeline not initialized - extraction results empty.")
        
        self.logger.info("Finished extracting info from prompt")
        return extracted_data