import numpy
import sklearn.feature_extraction
import sklearn.metrics
import spacy
import sklearn


nlp = spacy.load("en_core_web_trf")
nlp.add_pipe('sentencizer')

with open('../data/stopwords-en.txt', "r") as f:
    stop_words = f.readlines()
    stop_words = [word.strip() for word in stop_words]


def extract_top_n(text: str, interest: str, n: int) -> str:
    document = nlp(text)
    lemma_sentences = [sentence.lemma_ for sentence in document.sents]

    tokenized_interest = nlp(interest)
    clear_interest = " ".join([token.lemma_ for token in tokenized_interest if token.lemma_ not in stop_words])
    
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
    documents_tfidf = vectorizer.fit_transform(lemma_sentences)
    query_tfidf = vectorizer.transform([clear_interest])

    similarities = sklearn.metrics.pairwise.cosine_similarity(query_tfidf, documents_tfidf).flatten()

    documents_array = numpy.array(list(document.sents), dtype=object)
    top_indices = numpy.argpartition(similarities, -n)[-n:]
    top_documents = documents_array[top_indices]

    return " ".join(map(lambda sentence : sentence.text, top_documents))


if __name__ == "__main__":
    with open("../data/scraped_data/wiki_out_.json", "r") as f:
        text = f.read()
    
    interest = "Who wrote the screenplay for the Godfather?"
    extracted = extract_top_n(text, interest, 5)
    print(extracted)