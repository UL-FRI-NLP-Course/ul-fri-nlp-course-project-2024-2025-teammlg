import spacy
import warnings
warnings.filterwarnings('ignore')

class POStagger:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_trf")
        except:
            print("en_core_web_trf load failed!")
            self.nlp = None

    def tag(self, query: str):
        if self.nlp:
            document = self.nlp(query)

            out = {}
            for entity in document.ents:
                out[entity.text] = entity.label_
            return out

        else:
            return None
        
    #TODO multiple models + voting scheme??

