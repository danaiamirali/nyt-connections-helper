from operator import itemgetter
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import tool, AgentExecutor
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain_core.messages import SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain



from gensim.models import Word2Vec
import numpy as np
from Levenshtein import distance
from sklearn.cluster import KMeans

from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")


"""
Get the list of words in the game from the file.
"""
def get_words(filename: str) -> list:
    with open(filename, "r") as file:
        text = file.read()

    words = text.split(',')

    return [word.strip().lower() for word in words]



words = get_words("input.txt")
if __name__ == "__main__":
    llm = ChatOpenAI(api_key=API_KEY, model="gpt-4", temperature=0.5)
    memory = ConversationBufferMemory(memory_key="chat_history")

    system_message = f"""
    You are a chatbot helper for the word game "connections," where, provided a list of words, you must find the connection between them. The game is very difficult, and often the connections are significantly more complicated than appears at first glance.
    
    Typically, there are 16 words, and four groups of four words that are connected in some way. The goal of the game is to find the connection between the words and group them accordingly.

    Your task is to help the user. Do not solve the answer out right, but aid them based on their requests and responses. Help the user out with AT MOST one group of words at a time.

    You have several tools at your disposal, which you will have to take advantage of to properly help the user and figure out the connections yourself:
    - You can leverage some functions for:
        - You can find the levenstein edit distance between words.
        - You can obtain clusters of words based on their semantic meaning.
        - You can find other words that are linked to several of the words in the list.
    - You can leverage some external resources for:
        - You can ask the user for more information, such as what they think the connection is, or what they've already tried.
    
    The words are:
    {words}
    """


    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=system_message
            ),  # The persistent system prompt
            MessagesPlaceholder(
                variable_name="chat_history"
            ),  # Where the memory will be stored.
            HumanMessagePromptTemplate.from_template(
                "{human_input}"
            ),  # Where the human input will injected
        ]
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chat_llm_chain = LLMChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=True
    )

    result = chat_llm_chain.invoke("Provide the user a warm greeting.")
    print("Helper:", result["chat_history"][-1].content)

    while True:
        human_input = input("User: ")

        if human_input == "/exit":
            exit()
        
        result = chat_llm_chain.invoke(human_input)
        print("Helper:", result["chat_history"][-1].content)
