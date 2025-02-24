import gradio as gr

def chatbot_reply(user_message, history):
    # history is added so that the chat is always context aware
    ai_reply = "please wait till I implement this method "

    history.append([user_message, ai_reply])
    return "", history, history


with gr.Blocks(theme='gstaff/sketch') as demo:
    gr.Markdown(
        """
        # Legal RAG Chatbot
        Ask questions about the Supreme Court case!  
        This chatbot leverages rhetorical roles to retrieve the most relevant context 
        before generating an answer.
        """
    )

    state = gr.State([])

    chatbot = gr.Chatbot(label="Legal QA Chat supported by Rhetorical Roles")
    user_input = gr.Textbox(
        show_label=False,
        placeholder="Your Question here",
    )

    user_input.submit(fn=chatbot_reply, inputs=[user_input, state], outputs=[user_input, state, chatbot])


demo.launch(server_name="0.0.0.0", server_port=7860)
