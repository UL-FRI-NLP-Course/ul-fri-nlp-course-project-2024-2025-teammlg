import nltk

nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("maxent_ne_chunker_tab")
nltk.download("words")


def extract_names(query: str):
    tokens = nltk.word_tokenize(query)
    tags = nltk.pos_tag(tokens)
    chunks = nltk.ne_chunk(tags)

    for chunk in chunks:
        print(chunk)
        if type(chunk) == nltk.Tree:
            print(chunk)


if __name__ == "__main__":
    query = "Did Steven Spielberg shoot the film The Fablemans in 2022?"
    extract_names(query)
    # Classifies "Did" as a noun (and a person)
