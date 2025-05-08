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

            out_separated = {"movies": [], "people": [], "key": []}
            for entity in document.ents:

                if entity.label_ == 'PERSON':
                    out_separated["people"].append(entity.text)
                else: #TODO maybe something more intelligent here
                    #also it's not entirely stupid to also put PERSON in movies - sometimes you get some obscure documentary with some potential info
                    out_separated["movies"].append(entity.text)

            return out_separated

        else:
            return None
        
    #TODO multiple models + voting scheme??

