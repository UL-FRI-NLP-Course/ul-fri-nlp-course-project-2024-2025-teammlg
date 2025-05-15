"""import asyncio
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
print(score)"""

user_input = "Can you summarize the main themes of The Dark Knight for me?"
contexts = ["The Dark Knight from the year 2008 (Christopher Nolan). Batman raises the stakes in his war on crime. With the help of Lt. Jim Gordon and District Attorney Harvey Dent, Batman sets out to dismantle the remaining criminal organizations that plague the streets. The partnership proves to be effective, but they soon find themselves prey to a reign of chaos unleashed by a rising criminal mastermind known to the terrified citizens of Gotham as the Joker. A gang of masked criminals rob a mafia-owned bank in Gotham City, betraying and killing each other until the sole survivor, the Joker, reveals himself as the mastermind and escapes with the money. The vigilante Batman, district attorney Harvey Dent, and police lieutenant Jim Gordon ally to eliminate Gotham's organized crime. Batman's true identity, the billionaire Bruce Wayne, publicly supports Dent as Gotham's legitimate protector, believing Dent's success will allow him to retire as Batman and romantically pursue his childhood friend Rachel Dawes—despite her being with Dent. Gotham's mafia bosses gather to discuss protecting their organizations from the Joker, the police, and Batman. The Joker interrupts the meeting and offers to kill Batman for half of the fortune their accountant, Lau, concealed before fleeing to Hong Kong to avoid extradition. With the help of Wayne Enterprises CEO Lucius Fox, Batman finds Lau in Hong Kong and returns him to the custody of Gotham police. His testimony enables Dent to apprehend the crime families. The bosses accept the Joker's offer, and he kills high-profile targets involved in the trial, including the judge and police commissioner. Although Gordon saves the mayor, the Joker threatens that his attacks will continue until Batman reveals his identity. He targets Dent at a fundraising dinner and throws Rachel out of a window, but Batman rescues her. Wayne struggles to understand the Joker's motives, to which his butler Alfred Pennyworth says \"some men just want to watch the world burn.\" Dent claims he is Batman to lure out the Joker, who attacks the police convoy transporting him. Batman and Gordon apprehend the Joker, and Gordon is promoted to commissioner. At the police station, Batman interrogates the Joker, who says he finds Batman entertaining and has no intention of killing him. Having deduced Batman's feelings for Rachel, the Joker reveals she and Dent are being held separately in buildings rigged to explode. Batman races to rescue Rachel while Gordon and the other officers go after Dent, but they discover the Joker gave their positions in reverse. The explosives detonate, killing Rachel and severely burning Dent's face on one side. The Joker escapes custody, extracts the fortune's location from Lau, and burns it, killing Lau in the process. Coleman Reese, a consultant for Wayne Enterprises, deduces and tries to expose Batman's identity, but the Joker threatens to blow up a hospital unless Reese is killed. While the police evacuate hospitals and Gordon struggles to keep Reese alive, the Joker meets with a disillusioned Dent, persuading him to take the law into his own hands and avenge Rachel. Dent defers his decision-making to his now half-scarred, two-headed coin, killing the corrupt officers and the mafia involved in Rachel's death. As panic grips the city, the Joker reveals two evacuation ferries, one carrying civilians and the other prisoners, are rigged to explode at midnight unless one group sacrifices the other. To the Joker's disbelief, the passengers refuse to kill one another. Batman subdues the Joker but refuses to kill him. Before the police arrest the Joker, he says although Batman proved incorruptible, his plan to corrupt Dent has succeeded. Dent takes Gordon's family hostage, blaming his negligence for Rachel's death. He flips his coin to decide their fates, but Batman tackles him to save Gordon's son, and Dent falls to his death. Believing Dent is the hero the city needs and the truth of his corruption will harm Gotham, Batman takes the blame for his death and actions and persuades Gordon to conceal the truth. Pennyworth burns an undelivered letter from Rachel to Wayne that says she chose Dent, and Fox destroys the invasive surveillance network that helped Batman find the Joker. The city mourns Dent as a hero, and the police launch a manhunt for Batman."]
response="\n\n\"The Dark Knight\" explores several key themes:\n\n1. **Hero-Visa Duality**: Jack Nicholson, as the Dark Knight, seeks to protect others but ends up being harmed by a criminal gang that resembles the \"Kingsmen of the Dead.\"\n\n2. **Identity and Manipulation**: The film delves into how Jack sees himself\u2014both hero and villain\u2014and his actions sometimes seem manipulative.\n\n3. **Family Dynamics**: There's a tension between Jack and Harold Hurwitz, where Jack tries to protect others but may exploit their strength.\n\n4. **Power and Control**: The Gang's influence is both a threat and a source of strength for Jack, highlighting power dynamics within the narrative.\n\n5. **Social Commentary**: Despite its humor, \"The Dark Knight\" also touches on broader social issues, offering a critical look at the corrupting effects of certain behaviors.\n\nThese themes together create a complex exploration of Jack's multifaceted journey and his place in a world that feels both familiar and familiarized with his past."
ground_truth = "The Dark Knight (2008), directed by Christopher Nolan, explores several deep and interconnected themes. The main ones include:\n\n\t1. Chaos vs. Order: The Joker represents chaos, anarchy, and unpredictability, while Batman and the authorities strive to maintain order. The film explores how fragile societal structures are when challenged by extreme forces.\n\n\t2. Moral Ambiguity and Duality: The film questions traditional notions of good and evil. Batman must bend ethical lines to fight crime, while Harvey Dent’s transformation into Two-Face shows how easily a hero can fall.\n\n\t3. Justice vs. Vengeance: Batman operates outside the law but seeks justice, whereas characters like Dent blur the line by giving in to vengeance when wronged.\n\n\t4. The Nature of Heroism: The film challenges the idea of what makes someone a hero. Batman chooses to be seen as a villain to protect Gotham's hope, highlighting the theme of self-sacrifice for the greater good.\n\n\t5. Fear and Corruption: Fear is used as both a weapon and a shield, while Gotham's institutions are portrayed as vulnerable to corruption—something both Batman and the Joker exploit in different ways."

import os
os.environ["OPENAI_API_KEY"] = "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"

import deepeval
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from deepeval import evaluate

# gpt-4.1-mini

correctness_metric = GEval(
    name="Correctness",
    criteria="Check whether the facts in 'actual output' contradicts any facts in 'expected output'. If the actual output omits some facts, it's okay as long as it doesn’t contradict or distort the expected facts. If the task is to recommend or suggest items (e.g., movies), do not check for exact matches with expected output, instead check if the recommendations are similar in genre, theme, tone, or relevance, per the prompt.",
    model="gpt-3.5-turbo",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    verbose_mode=True
)

test_case = LLMTestCase(
    input=user_input,
    actual_output=response,
    expected_output=ground_truth,
    retrieval_context=contexts
)

print(evaluate(test_cases=[test_case], metrics=[correctness_metric]))


