import spacy
import pytextrank
import warnings

warnings.filterwarnings("ignore")


class Summarizer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")
        # Adding biased text summarization via pytextrank
        self.nlp.add_pipe("biasedtextrank")

    def summarize(self, file, entity_of_interest):
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
            return out
