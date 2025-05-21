from abc import ABC, abstractmethod
import enum
import logging
from typing import List, TypedDict

import numpy
import spacy
import sklearn
import sklearn.feature_extraction
import sklearn.metrics


class ExtractedData(TypedDict):
    movies: List[str]
    people: List[str]
    other: List[str]


class DocumentExtractionMethod(enum.Enum):
    Null = enum.auto()
    All = enum.auto()
    TFIDF = enum.auto()


class Extractor(ABC):
    @abstractmethod
    def extract_useful_information(self, user_prompt: str) -> ExtractedData:
        ...


class NullPromptExtractor(Extractor):
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


class AdvancedExtractor(Extractor):
    def __init__(
        self,
        label: str = "EntityExtractor",
        accurate_model: bool = False,
        logging_level: int = logging.INFO
    ):
        super().__init__()
        self.logger = logging.getLogger(label)
        self.logger.setLevel(logging_level)
        self.logger.addHandler(logging.StreamHandler())

        if not accurate_model:
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
        
        if self.nlp is not None:
            self.nlp.add_pipe('sentencizer')
        
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
    
    def extract_from_documents(self, documents: List[str], interest: List[str], method: DocumentExtractionMethod, n_sentences: int = 5) -> List[str]:
        if method == DocumentExtractionMethod.TFIDF:
            return self.extract_sentences_with_tfidf(documents, interest, n_sentences)
        elif method == DocumentExtractionMethod.All:
            return documents
        return []

    def extract_sentences_with_tfidf(self, documents: List[str], interest: List[str], n: int) -> List[str]:
        documents_text = " ".join(documents)
        documents_text = documents_text.replace("\n", ". ")

        interest_text = " ".join(interest)
        interest_text = interest_text.replace("\n", ". ")

        tokenized_document = self.nlp(documents_text)
        tokenized_interest = self.nlp(interest_text)

        document_lemma_sentences = [sentence.lemma_ for sentence in tokenized_document.sents]
        clear_interest = " ".join([token.lemma_ for token in tokenized_interest if token.lemma_ not in self.stop_words])

        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
        documents_tfidf = vectorizer.fit_transform(document_lemma_sentences)
        query_tfidf = vectorizer.transform([clear_interest])

        similarities = sklearn.metrics.pairwise.cosine_similarity(query_tfidf, documents_tfidf).flatten()
        
        documents_array = numpy.array(list(tokenized_document.sents), dtype=object)
        top_indices = numpy.argsort(similarities)[-n][::-1]
        top_documents = documents_array[top_indices].tolist()

        return top_documents
