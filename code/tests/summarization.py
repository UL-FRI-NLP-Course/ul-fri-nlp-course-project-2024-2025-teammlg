import spacy
import pytextrank  # Don't remove this! Otherwise, spacy doesn't recognize 'biasedtextrank' in the pipeline

nlp = spacy.load("en_core_web_trf")
# Adding biased text summarization via pytextrank
nlp.add_pipe("biasedtextrank")


def summarization_test(file: str, entity_of_interest: str):
    with open(file, "r") as f:
        document = f.read()

    document = nlp(document)
    textrank = document._.textrank
    # We force it to be biased towards our entity of interest
    textrank.change_focus(entity_of_interest, bias=10.0, default_bias=0.0)

    # Basically just takes the sentences it deems statistically to best represent the context
    # and returns them
    for sentence in textrank.summary(limit_phrases=15, limit_sentences=5):
        print(sentence, end=" ")
    print()


def summarization_test_02(file: str, entity_of_interest: str):
    with open(file, "r") as f:
        document = f.read()

    # A bit of cleanup
    document = document.replace("\n", ". ")
    document = document.replace("\t", ". ")
    document = document.replace("\r", "")

    document = nlp(document)
    textrank = document._.textrank
    # We force it to be biased towards our entity of interest
    textrank.change_focus(entity_of_interest, bias=10.0, default_bias=0.0)

    # Basically just takes the sentences it deems statistically to best represent the context
    # and returns them
    for sentence in textrank.summary(limit_phrases=15, limit_sentences=5):
        print(sentence, end=" ")
    print()
    print()


if __name__ == "__main__":
    summarization_test_02("./war_of_the_worlds_wiki.txt", "Spielberg")
    # Performs poorly because of how unstructured the text is (I just copy-pasted Wikipedia page, without formatting it).
    # It does retain some information about the film (but focuses on John Williams for some reason???)

    summarization_test("./christmas_carol.txt", "Ebenezer Scrooge")
    # Focuses on the correct subject, but remains disjointed.
    # Cannot function well as a summary, but may provide useful context for the LLM.

    # This stuff is somewhat slow ... Can we speed it up?

    # Some reading on text summarization and information extraction:
    #  - https://dl.acm.org/doi/pdf/10.1145/3545176
    #  - https://dl.acm.org/doi/pdf/10.1145/3529754
    #  - https://arxiv.org/pdf/2504.13054 (This one could be interesting to test)
