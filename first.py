from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
llm=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
checkpointer=InMemorySaver()

agent=create_agent(llm,tools=[],system_prompt=(
        "You are a helpful AI assistant. "
        "Remember previous messages in the conversation. "
        "Answer questions using conversation context."
    ),checkpointer=checkpointer)

session_id = "user_1"
config = {
    "configurable": {
        "thread_id": session_id
    }
}
while True:
    user_input=input("You:")
    if user_input.lower()=="exit":
        break
    response=agent.invoke({"messages": [{
                        "role": "user",
                        "content": user_input }]},config=config)
    ai_response = response["messages"][-1].content
    print("AI:", ai_response)