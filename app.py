import os
from openai import OpenAI
import gradio as gr
from dotenv import load_dotenv

load_dotenv() 

client = OpenAI()

MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

SYSTEM_PROMPT = (
    "You are a helpful, friendly assistant. Explain steps clearly and concisely. "
    "If the user asks for code or commands, present them in code blocks."
)

def _openai_chat(messages):

    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.6,
    )
    return resp.choices[0].message["content"]

def respond(user_message, chat_history):
 
    chat_history = chat_history or []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, b in chat_history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": b})
    messages.append({"role": "user", "content": user_message})
    assistant_text = _openai_chat(messages)
    chat_history.append((user_message, assistant_text))
    return "", chat_history

def reset_chat():
    return []

def summarize_and_append(text, chat_history):
    
    if not text:
        return gr.update(value=chat_history)
    messages = [
        {"role": "system", "content": "You are a concise summarizer."},
        {"role": "user", "content": f"Summarize the following text in 6 bullet points:\n\n{text}"}
    ]
    summary = _openai_chat(messages)
    chat_history = chat_history or []
    chat_history.append(("(uploaded text)", summary))
    return chat_history

with gr.Blocks() as demo:
    gr.Markdown("# OpenAI Agent \nOPENAI chat agent ")
    chatbot = gr.Chatbot()
    with gr.Row():
        txt = gr.Textbox(placeholder="Type your message and press Enter", show_label=False)
        clear_btn = gr.Button("Clear")
    with gr.Accordion("Tools", open=False):
        summary_input = gr.Textbox(label="Paste text to summarize", lines=5, placeholder="Paste long text here...")
        summary_btn = gr.Button("Summarize & append to chat")

    txt.submit(respond, [txt, chatbot], [txt, chatbot])
    clear_btn.click(reset_chat, [], chatbot)
    summary_btn.click(summarize_and_append, [summary_input, chatbot], chatbot)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))

