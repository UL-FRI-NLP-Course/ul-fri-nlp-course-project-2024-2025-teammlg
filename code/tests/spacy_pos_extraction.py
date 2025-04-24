import spacy
#import spacy_transformers

nlp = spacy.load("en_core_web_trf")


def extract_names(query: str):
    document = nlp(query)

    print(f"Entities for {query}:")
    for entity in document.ents:
        print(f"\t{entity.text} ({entity.label_})")


if __name__ == "__main__":
    query = "Did Steven Spielberg shoot the film The Fablemans in 2022?"
    extract_names(
        query
    )  # Correctly finds Stephen Spielberg and The Fabelmans, and that 2022 is a date

    query = "What about Jaws, did he do it in the 80s?"
    extract_names(query)  # Correctly finds Jaws, and that the 80s are a date

    query = "Do you know anything about the film I Know What You Did Last Summer?"
    extract_names(query)  # Correctly finds the movie

    query = "Who plays in the movie Conclave?"
    extract_names(query)  # Finds Conclave
    query = "Who acts in conclave?"
    extract_names(query)  # Does not find anything
    query = "Who acts in the conclave?"
    extract_names(query)  # Does not find anything
    query = "Who plays in conclave?"
    extract_names(query)  # Does not find anything
    query = "What genre is Mickey 17?"
    extract_names(query)  # Finds Mickey (person) and 17 (cardinal)
    query = "What genre is the film Mickey 17?"
    extract_names(query)  # Correctly classifies Mickey 17
    query = "When does The Amateur come to cinemas?"
    extract_names(query)  # Correctly classifies The Amateur
    query = "When does the amateur come to cinemas?"
    extract_names(query)  # Finds nothing
    query = "Is there any news of the film 'Oh, The Places You'll Go!'? I'm looking forward to it."
    extract_names(query)  # Finds the film, but only with quotation marks

    """OBSERVATIONS:
    - It sometimes helps if the title is spelled and capitalized correctly
    - Additional context helps: preceed the title with 'movie' or similar
    - Context helps (verbs shoot, film, act; phrases 'come to cinema', ...)
    - It helps to put the title in quotation marks

    - Overall better than nltk, but slightly slower
    - Currently problems on Python 3.13, cannot install with pip or conda
    - I tested this with local Python 3.10 installation
    """
