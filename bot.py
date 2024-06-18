import streamlit as st
from utils import write_message
from agent import generate_response

# tag::setup[]
# Page Config
st.set_page_config("Ebert", page_icon=":movie_camera:")
# end::setup[]

# tag::session[]
# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm the EU AI Act Chatbot!  How can I help you?"},
    ]
# end::session[]


# tag::submit[]
# Submit handler
def handle_submit(message):
    # Handle the response
    with st.spinner('Thinking...'):
        print("The message is a %s." % type(message))  # for debug
        print("Here is the message: %s" % message)
        response = generate_response(message)
        print("The response is a %s." % type(response))  # for debug
        print("Here is the response: %s" % response)
        write_message('assistant', response)
# end::submit[]


# tag::chat[]
# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    write_message('user', prompt)

    # Generate a response
    handle_submit(prompt)
# end::chat[]
