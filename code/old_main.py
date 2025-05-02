from typing import Iterator

import ollama
from models.deepseek.deepseek_front import DeepSeekFilmChatBot


def test_deepseek_chatbot():
    """A test on how DeepSeek works"""
    deepseek_chatbot = DeepSeekFilmChatBot()

    # It takes some time to respond, because we do not stream text to console
    response1 = deepseek_chatbot.prompt("Hi, how are you?")
    print(response1.response)

    # This one may take particularly long
    response2 = deepseek_chatbot.prompt("What is a film?")
    print(response2.response)


def clean_up_prompt(prompt: str) -> str:
    """Receives the prompt and performs operations to clean it up (trim whitespace at start/end, etc.)"""
    prompt = prompt.strip()
    return prompt


def conversation():
    """A conversation loop with chatbot"""

    prompt_indicator = "> "  # Indicates that the user can input text

    deepseek_chatbot = DeepSeekFilmChatBot(
        "deepseek-r1:1.5b", "models/deepseek", "data/scraped_data", mode="advanced"
    )
    user_prompt = clean_up_prompt(input(prompt_indicator))
    while user_prompt != "quit":
        # TODO: Extract useful information
        # TODO: Retrieve information from the internet
        # TODO: Process information
        data = ""
        responses: Iterator[ollama.GenerateResponse] = deepseek_chatbot.prompt_stream(
            user_prompt, data=data
        )
        is_thinking = True
        print("Thinking...")
        for i, response in enumerate(responses):
            if not is_thinking:
                print(response.response, end="", flush=True)
            if response.response == "</think>":
                is_thinking = False
        print()

        user_prompt = clean_up_prompt(input(prompt_indicator))  # For next iteration


if __name__ == "__main__":
    # test_deepseek_chatbot()
    conversation()
