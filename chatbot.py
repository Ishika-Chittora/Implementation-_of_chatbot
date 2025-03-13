import streamlit as st
import random
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from datetime import datetime

# Set page config (MUST BE THE FIRST STREAMLIT COMMAND)
st.set_page_config(
    page_title="Chatbot",
    page_icon="🌍",
    layout="wide"
)

# Load intents from the JSON file
with open('intents.json', 'r') as file:
    intents_data = json.load(file)

intents = intents_data['intents']

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Extract patterns and tags for training
tags = []
patterns = []

for intent in intents:
    tag = intent['tag']
    for pattern in intent['patterns']:
        tags.append(tag)
        patterns.append(pattern)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess and vectorize the patterns
X = vectorizer.fit_transform(patterns)
y = tags

# Train the model
clf.fit(X, y)

# Chatbot function
def chatbot(input_text):
    if not input_text.strip():
        return "Please type something so I can help you!"
    
    # Preprocess user input
    input_text = preprocess_text(input_text)
    
    # Vectorize the input text
    input_text_vec = vectorizer.transform([input_text])
    
    # Predict the intent
    tag = clf.predict(input_text_vec)[0]
    
    # Find the corresponding intent and return a random response
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    
    # Default response if no intent is matched
    return "I'm sorry, I didn't understand that. Can you please rephrase?"

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
    }
    /* Chat response box */
    .response-box {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    /* Button styling */
    .stButton button {
        background-color: #ff6f61;
        color: white;
        border-radius: 20px;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #ff3b2f;
    }
    /* Input box styling */
    .stTextInput input {
        border-radius: 20px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation
st.sidebar.title("Chatbot - just chatter")
menu = st.sidebar.radio("Navigate", ["Chat", "Conversation History", "About"])

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat Page
if menu == "Chat":
    st.title("💬 Chat with me")
    st.markdown("Ask me anything!")

    # User input
    user_input = st.text_input("You: ", placeholder="Type your message here...", key="user_input")

    # Send button
    if st.button("Send"):
        if user_input.strip():
            # Get chatbot response
            bot_response = chatbot(user_input)
            
            # Add user input and bot response to chat history
            st.session_state.chat_history.append({"role": "user", "message": user_input, "time": datetime.now().strftime("%H:%M")})
            st.session_state.chat_history.append({"role": "bot", "message": bot_response, "time": datetime.now().strftime("%H:%M")})

    # Display the latest bot response in a creative box
    if st.session_state.chat_history:
        latest_response = st.session_state.chat_history[-1]["message"]
        st.markdown(
            f'<div class="response-box"><strong>ChatBot:</strong> {latest_response}</div>',
            unsafe_allow_html=True
        )

# Conversation History Page
elif menu == "Conversation History":
    st.title("📜 Conversation History")
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.markdown(
                    f'<div class="response-box"><strong>You:</strong> {chat["message"]}<br><small>{chat["time"]}</small></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="response-box"><strong>ChatBot:</strong> {chat["message"]}<br><small>{chat["time"]}</small></div>',
                    unsafe_allow_html=True
                )
    else:
        st.write("No conversation history yet.")

# About Page
elif menu == "About":
    st.title("A chatbot for chatter folks")
    st.markdown("""The chatbot interface is built using Streamlit. The interface includes a text input box for users to input their text and a chat window to display the chatbot's responses. The interface uses the trained model to generate responses to user input.
    """)
    
    # Display images with effects
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://miro.medium.com/max/1024/1*C7kjZxHFJ1k64SSoXcHjhQ.jpeg", use_column_width=True)