from pipeline import ChatbotPipeline


def converse(pipeline: ChatbotPipeline):
    print("You will be prompted to type after a '>' symbol. You can finish your conversation by typing 'quit'.")
    user_input = input("> ")

    while user_input != "quit":
        pipeline_result = pipeline.run(user_input)
        print(pipeline_result["generated_response"])
        user_input = input(">")