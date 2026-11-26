from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
import os

load_dotenv()
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
llm=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
chat_history=[SystemMessage(content="You are a python tutor")]
WINDOW_SIZE=2
while True:
    user_input=input("You:")
    if user_input.lower()=="exit":
        break
    chat_history.append(HumanMessage(content=user_input))
    response=llm.invoke(chat_history)
    print("AI :",response.content)
    chat_history.append(AIMessage(content=response.content))
    if len(chat_history) > (1 + WINDOW_SIZE * 2):
        chat_history = [chat_history[0]] + chat_history[-(WINDOW_SIZE * 2):]