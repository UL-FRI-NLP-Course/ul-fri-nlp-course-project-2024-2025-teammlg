import json
import time
import numpy
import sklearn.feature_extraction
import sklearn.metrics
import spacy
import sklearn


nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('sentencizer')

with open('../data/stopwords-en.txt', "r") as f:
    stop_words = f.readlines()
    stop_words = [word.strip() for word in stop_words]


def extract_top_n(text: str, interest: str, n: int) -> list[str]:
    start_time = time.perf_counter()

    document = nlp(text)
    lemma_sentences = [sentence.lemma_ for sentence in document.sents]

    print(f"{time.perf_counter() - start_time} s needed for document parsing")

    tokenized_interest = nlp(interest)
    clear_interest = " ".join([token.lemma_ for token in tokenized_interest if token.lemma_ not in stop_words])

    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
    documents_tfidf = vectorizer.fit_transform(lemma_sentences)
    query_tfidf = vectorizer.transform([clear_interest])

    similarities = sklearn.metrics.pairwise.cosine_similarity(query_tfidf, documents_tfidf).flatten()

    documents_array = numpy.array(list(document.sents), dtype=object)
    top_indices = numpy.argsort(similarities)[-n:][::-1]
    top_documents = documents_array[top_indices]

    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time} s")

    return " ".join(map(lambda sentence : sentence.text, top_documents))


if __name__ == "__main__":
    with open("../data/scraped_data/wiki_out_.json", "r") as f:
        data = json.load(f)
        text = " "
        for key, value in data.items():
            value = value.get(key, None)
            if value is None:
                continue
            text += value["summary"]
            text += " "
            text += value["plot"]
            text += " "
    
    interest = "Who wrote the screenplay for the Godfather?"
    extracted = extract_top_n(text, interest, 3)
    print(extracted)