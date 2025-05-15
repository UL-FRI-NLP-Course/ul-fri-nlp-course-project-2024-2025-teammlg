import asyncio
from ragas import SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
import ragas.metrics as ragasMetric
from langchain_openai import ChatOpenAI

print("start")

user_input = "Can you summarize the main themes of The Dark Knight for me?"
contexts = ["Batman raises the stakes in his war on crime. With the help of Lt. Jim Gordon and District Attorney Harvey Dent, Batman sets out to dismantle the remaining criminal organizations that plague the streets. The partnership proves to be effective, but they soon find themselves prey to a reign of chaos unleashed by a rising criminal mastermind known to the terrified citizens of Gotham as the Joker."]
response="\n\n\"The Dark Knight\" explores several key themes:\n\n1. **Hero-Visa Duality**: Jack Nicholson, as the Dark Knight, seeks to protect others but ends up being harmed by a criminal gang that resembles the \"Kingsmen of the Dead.\"\n\n2. **Identity and Manipulation**: The film delves into how Jack sees himself\u2014both hero and villain\u2014and his actions sometimes seem manipulative.\n\n3. **Family Dynamics**: There's a tension between Jack and Harold Hurwitz, where Jack tries to protect others but may exploit their strength.\n\n4. **Power and Control**: The Gang's influence is both a threat and a source of strength for Jack, highlighting power dynamics within the narrative.\n\n5. **Social Commentary**: Despite its humor, \"The Dark Knight\" also touches on broader social issues, offering a critical look at the corrupting effects of certain behaviors.\n\nThese themes together create a complex exploration of Jack's multifaceted journey and his place in a world that feels both familiar and familiarized with his past."
ground_truth = "The Dark Knight (2008), directed by Christopher Nolan, explores several deep and interconnected themes. The main ones include:\n\n\t1. Chaos vs. Order: The Joker represents chaos, anarchy, and unpredictability, while Batman and the authorities strive to maintain order. The film explores how fragile societal structures are when challenged by extreme forces.\n\n\t2. Moral Ambiguity and Duality: The film questions traditional notions of good and evil. Batman must bend ethical lines to fight crime, while Harvey Dent’s transformation into Two-Face shows how easily a hero can fall.\n\n\t3. Justice vs. Vengeance: Batman operates outside the law but seeks justice, whereas characters like Dent blur the line by giving in to vengeance when wronged.\n\n\t4. The Nature of Heroism: The film challenges the idea of what makes someone a hero. Batman chooses to be seen as a villain to protect Gotham's hope, highlighting the theme of self-sacrifice for the greater good.\n\n\t5. Fear and Corruption: Fear is used as both a weapon and a shield, while Gotham's institutions are portrayed as vulnerable to corruption—something both Batman and the Joker exploit in different ways."

sample = SingleTurnSample(
    user_input=user_input,
    retrieved_contexts=contexts,
    response=response,
    reference=ground_truth
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm = LangchainLLMWrapper(llm)
#messages = [HumanMessage(content="What's the weather like today?")]
#response = llm.invoke(messages)
#print(response.content)

faithfulness = ragasMetric.FactualCorrectness(llm=llm)
score = asyncio.run(faithfulness.single_turn_ascore(sample))
print(score)