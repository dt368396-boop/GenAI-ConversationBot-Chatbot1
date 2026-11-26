from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
import gradio as gr
import uuid
import os

load_dotenv()
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
llm=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
checkpointer=InMemorySaver()
agent = create_agent(model=llm,tools=[],
    system_prompt="""
    You are a helpful AI teaching assistant.

    Answer questions clearly and explain concepts
    with examples wherever necessary.

    Remember previous messages in the conversation.
    """,
    checkpointer=checkpointer
)

def new_chat():
    session_id=str(uuid.uuid4())
    print(session_id)
    return [],session_id

def chatbot_response(user_input,chatbot,session_id):
    config = {
        "configurable": {
            "thread_id": session_id
        }
    }
    response=agent.invoke({"messages": [{
                        "role": "user",
                        "content": user_input }]},config=config)
    answer=response["messages"][-1].content
    chatbot = chatbot or []
    chatbot.append({"role": "user", "content": user_input})
    chatbot.append({"role": "assistant", "content": answer})
    return "",chatbot


with gr.Blocks(title="LangChain Conversational Chatbot") as demo:
    gr.Markdown("# LangChain Conversational Chatbot")
    gr.Markdown(
        "Chat with an AI assistant that remembers "
        "your previous messages."
    )
    session_id = gr.State(value=lambda: str(uuid.uuid4()))
    chatbot = gr.Chatbot(height=450, label="AI Assistant")
    user_input = gr.Textbox(placeholder="Ask me anything...", label="Your Message")
    with gr.Row():
        send_btn = gr.Button("Send",variant="primary")
        new_chat_btn = gr.Button("New Chat")

    send_btn.click(fn=chatbot_response,inputs=[user_input,chatbot,session_id],outputs=[
            user_input,chatbot])
    user_input.submit(fn=chatbot_response, inputs=[user_input, chatbot, session_id], outputs=[
        user_input, chatbot])
    new_chat_btn.click(fn=new_chat, inputs=[], outputs=[chatbot, session_id])

import os
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))