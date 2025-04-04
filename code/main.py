from deepseek_front import DeepSeekChatBot


def test_deepseek_chatbot():
    deepseek_chatbot = DeepSeekChatBot()

    # It takes some time to respond, because we do not stream text to console
    response1 = deepseek_chatbot.prompt("Hi, how are you?")
    print(response1.response)

    # This one may take particularly long
    response2 = deepseek_chatbot.prompt("What is a film?")
    print(response2.response)


if __name__ == "__main__":
    test_deepseek_chatbot()
