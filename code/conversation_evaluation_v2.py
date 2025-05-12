import nltk.translate.bleu_score
import rouge_score.rouge_scorer
from models.deepseek.deepseek_front import DeepSeekFilmChatBot
import nltk
import rouge_score
import json


class FinalScore:
    rouge_1: float
    rouge_2: float
    rouge_5: float
    rouge_l: float
    bleu: float
    f1: float

    def __init__(self):
        self.rouge_1 = 0
        self.rouge_2 = 0
        self.rouge_5 = 0
        self.rouge_l = 0
        self.bleu = 0
        self.f1 = 0

    def __str__(self):
        return f"\tROUGE 1: {self.rouge_1}\n\tROUGE 2: {self.rouge_2}\n\tROUGE 5: {self.rouge_5}\n\tROUGE L: {self.rouge_l}\n\tBLEU: {self.bleu}\n\tF1: {self.f1}"


nltk.download("stopwords")


def clean_up_text(text: str) -> str:
    output_text: list[str] = []
    stop_words = set(nltk.corpus.stopwords.words("english"))
    word_tokens = nltk.tokenize.word_tokenize(text)
    is_thinking = True
    for token in word_tokens:
        if token == "</think>":
            is_thinking == False
        if is_thinking:
            continue
        output_text.append(token)

    output_text = [word for word in output_text if word not in stop_words]
    output_text = " ".join(output_text)
    return output_text


def test():
    scorer = rouge_score.rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    questions = []
    gpt_answers = []

    with open("data/evaluation_questions.json", "r") as f:
        j = json.loads(f.read())
        questions = [q[0] for q in j["scenarios"]]
        gpt_answers = [a[0] for a in j["ground_truth"]]

    deepseek = DeepSeekFilmChatBot(
        "deepseek-r1:1.5b", "models/deepseek", "data/scraped_data", mode="advanced"
    )
    deepseek_results = FinalScore()

    for question, gpt_answer in zip(questions, gpt_answers):
        print(question)
        print(gpt_answer)
        deepseek_response = deepseek.prompt_nonstream(question)
        print(deepseek_response)
        deepseek_response = clean_up_text(deepseek_response[0])
        gpt_response = clean_up_text(gpt_answer)
        # TODO: Calculate ROUGE-1/2/5/L score
        rouge = scorer.score(gpt_response, deepseek_response)
        # TODO: Calculate BLEU score
        weights = (0.25, 0.25, 0, 0)
        bleu = nltk.translate.bleu_score.sentence_bleu(
            gpt_response.split(" "), deepseek_response.split(" "), weights=weights
        )
        print(rouge)
        print(bleu)
        # TODO: Calculate F1 score
        pass

    print(f"Deepseek results:\n{deepseek_results}")


if __name__ == "__main__":
    test()
