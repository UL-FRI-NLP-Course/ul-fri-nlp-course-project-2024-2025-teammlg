import logging

def test_deepseek():
    logging.info("Testing transformers and DeepSeek....")
    try:
        import transformers
        model_label = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        temperature = 0.6
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_label)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_label,
            device_map="auto",
            torch_dtype="auto"
        )
        pad_token_id = tokenizer.eos_token_id
        chat = {
            "role": "user",
            "content": "Hi! How are you?"
        }
        text = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        input_tokens = tokenizer([text], return_tensors="pt").to('cuda')
        outputs = model.generate(
            **input_tokens,
            max_new_tokens=32768,
            pad_token_id=pad_token_id,
            temperature=temperature
        )
        final_output = ""
        for i in range(len(outputs)):
            output_text = tokenizer.decode(outputs[i])
            final_output += output_text
        logging.debug(f"DeepSeek output: {final_output}")
        logging.info("Test completed!\n")
    except Exception as e:
        logging.error(str(e))


def test_qwen():
    logging.info("Testing transformers and Qwen....")
    try:
        import transformers
        model_label = "Qwen/Qwen3-8B"
        temperature = 0.7
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_label)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_label,
            device_map="auto",
            torch_dtype="auto"
        )
        pad_token_id = tokenizer.eos_token_id
        chat = {
            "role": "user",
            "content": "Hi! How are you?"
        }
        text = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        input_tokens = tokenizer([text], return_tensors="pt").to('cuda')
        outputs = model.generate(
            **input_tokens,
            max_new_tokens=32768,
            pad_token_id=pad_token_id,
            temperature=temperature
        )
        final_output = ""
        for i in range(len(outputs)):
            output_text = tokenizer.decode(outputs[i])
            final_output += output_text
        logging.info(f"Qwen output: {final_output}")
        logging.info("Test completed!\n")
    except Exception as e:
        logging.error(str(e))


def test_entities_recognition_from_text():
    logging.info("Testing spacy and text entities....")
    try:
        import spacy
        nlp = spacy.load("en_core_web_trf")
        query = "Did Steven Spielberg shoot the film The Fablemans in 2022?"
        document = nlp(query)
        logging.debug(f"Parse result: {document.ents}")
        logging.info("Test completed!\n")
    except Exception as e:
        logging.error(f"{e}")


def test_summarization():
    logging.info("Testing summarization")
    try:
        import pytextrank
        import spacy

        with open("./war_of_the_worlds_wiki.txt", "r") as f:
            document = f.read()
        document = document.replace("\n", ". ").replace("\t", ". ").replace("\r", "")
        entity_of_interest = "Steven Spielberg"

        nlp = spacy.load("en_core_web_trf")
        nlp.add_pipe("biasedtextrank")
        
        document = nlp(document)
        textrank = document._.textrank
        textrank.change_focus(entity_of_interest, bias=10.0, default_bias=0.0)
        
        summary = ""
        for sentence in textrank.summary(limit_phrases=15, limit_sentences=5):
            summary += sentence
        logging.debug(f"Summary: {summary}")
        logging.info("Test completed!\n")
    except Exception as e:
        logging.error(f"{e}")


def test_evaluation():
    logging.info("Testing evaluation frameworks")
    try:
        import rouge_score
        import nltk
        import sklearn
        scorer = rouge_score.rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        rouge = scorer.score("I would like a piece of cheese which I so love.", "I love cheese and I would like a piece.")
        logging.debug(f"Rouge results: {rouge}")

        bleu = nltk.translate.bleu_score.sentence_bleu(
            "I would like a piece of cheese which I so love.", "I love cheese and I would like a piece.", weights=(0.25, 0.25, 0, 0)
        )
        logging.debug(f"Bleu score: {bleu}")
        logging.info("Test completed!\n")
    except Exception as e:
        logging.error(f"{e}")


def test_data_retrieval():
    logging.info("Testing data retrieval...")
    try:
        import bs4
        import requests
        import wikipedia
        logging.info("Test completed!\n")
    except Exception as e:
        logging.error(f"{e}")


if __name__ == "__main__":
    logging.basicConfig(filename="testing_log.log",
        filemode='a',
        format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG)
    logging.info("---------------------------------------")
    logging.info("STARTING TESTING PROCEDURE")
    logging.info("---------------------------------------")

    test_deepseek()
    test_qwen()
    test_entities_recognition_from_text()
    test_summarization()
    test_evaluation()
    test_data_retrieval()
