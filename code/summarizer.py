import spacy
#import pytextrank
import warnings
import json
import numpy
import sklearn.feature_extraction
import sklearn.metrics
import sklearn

warnings.filterwarnings("ignore")


with open('./data/stopwords-en.txt', "r") as f:
    stop_words = f.readlines()
    stop_words = [word.strip() for word in stop_words]


def json_to_text(j) -> str:
    if type(j) in [str, int, float]:
        return str(j)
    
    text = ""
    if type(j) is list:
        for i in j:
            text += json_to_text(i) + ", "
            
    if type(j) is dict:
        for key, value in j.items():
            text += str(key)
            text += ": "
            text += json_to_text(value)
            text += ". "
    return text


class Summarizer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        # Adding biased text summarization via pytextrank
        self.nlp.add_pipe('sentencizer')
    
    def extract_important(self, content: str, prompt: str, n: int = 5, is_json: bool = True) -> str:
        if content == "":
            return ""
        
        if is_json:
            content = json.loads(content)
            text = json_to_text(content)
        else:
            text = content

        document = self.nlp(text)
        lemma_sentences = [sentence.lemma_ for sentence in document.sents]
        if not lemma_sentences:
            return ""

        tokenized_interest = self.nlp(prompt)
        clear_interest = " ".join([token.lemma_ for token in tokenized_interest if token.lemma_ not in stop_words])

        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
        documents_tfidf = vectorizer.fit_transform(lemma_sentences)
        query_tfidf = vectorizer.transform([clear_interest])

        similarities = sklearn.metrics.pairwise.cosine_similarity(query_tfidf, documents_tfidf).flatten()

        documents_array = numpy.array(list(document.sents), dtype=object)
        #top_indices = numpy.argsort(similarities)[-n:][::-1]
        top_indices = numpy.argsort(similarities)[0:len(similarities)][::-1]
        top_documents = documents_array[top_indices].flatten()

        return " ".join(map(lambda sentence : sentence.text, top_documents))

    """def summarize(self, file, entity_of_interest):
        with open(file, 'r', encoding='UTF-8') as f:
            document = f.read()
            document = (
                document.replace("{", "")
                .replace("}", "")
                .replace("[", "")
                .replace("]", "")
            )

            document = self.nlp(document)
            textrank = document._.textrank
            # We force it to be biased towards our entity of interest
            textrank.change_focus(entity_of_interest, bias=10.0, default_bias=0.0)

            # Basically just takes the sentences it deems statistically to best represent the context
            # and returns them
            out = ""
            for sentence in textrank.summary(limit_phrases=15, limit_sentences=5):
                out += str(sentence)
            return out"""
