import streamlit as st
import openai

# Set your OpenAI API key
openai.api_key = "YOUR_API_KEY_HERE"

st.title("GPT-3 Chatbot")

# Define a function to interact with the GPT-3 API
def chat_with_gpt3(user_message):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=user_message,
        max_tokens=50,
        stop=None,
        temperature=0.7
    )
    return response.choices[0].text

# Create a text input box for the user to type messages
user_input = st.text_input("You:", value="")

if user_input:
    # Display the user's message
    st.write("You:", user_input)

    # Get a response from the chatbot
    bot_response = chat_with_gpt3(user_input)
    st.write("Chatbot:", bot_response)

# Note: Replace "YOUR_API_KEY_HERE" with your actual OpenAI API key

# Add a sidebar to explain how to use the chatbot
st.sidebar.header("Instructions")
st.sidebar.markdown(
    """
    1. Type your message in the text input box on the left.
    2. Press Enter or click the 'Submit' button.
    3. The chatbot will respond to your message.
    """
)

# Add a "Submit" button for user input
if st.button("Submit"):
    user_input = st.text_input("You:")
