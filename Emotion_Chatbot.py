import streamlit as st
from deepface import DeepFace
import groq 
import time
import cv2
import numpy as np

# Configure Streamlit page
st.set_page_config(
    page_title="Emotion-Based AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize Groq client with API key from Streamlit secrets
try:
    groq_api_key = st.secrets["groq_api_key"]
except:
    # Fallback for local development
    groq_api_key = "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"
client = groq.Client(api_key=groq_api_key)

def detect_emotion():
    """
    Detect emotion from camera input using DeepFace
    """
    # Use Streamlit's camera input
    camera_input = st.camera_input("Take a photo")
    
    if camera_input is not None:
        try:
            # Convert the image to bytes
            bytes_data = camera_input.getvalue()
            # Convert to numpy array
            file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
            # Decode the image
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Analyze emotion using DeepFace
            with st.spinner("Detecting emotion..."):
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                return emotion
        except Exception as e:
            st.error(f"Error detecting emotion: {str(e)}")
            return None
    return None

def generate_ai_response(user_message, emotion):
    """
    Generate AI response based on user message and detected emotion
    """
    try:
        prompt = f"""The user is feeling {emotion}. They said: '{user_message}'. 
        Respond appropriately and empathetically in 2-3 sentences."""
        
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": "You are a helpful and empathetic AI assistant. Keep responses concise and focused on the user's emotional state."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating AI response: {str(e)}")
        return "I apologize, but I'm having trouble generating a response right now. Please try again."

def initialize_session_state():
    """
    Initialize session state variables
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "emotion" not in st.session_state:
        st.session_state.emotion = None

def display_chat_history():
    """
    Display chat history with styling
    """
    st.write("### Chat History")
    for message in st.session_state.chat_history:
        if message.startswith("You: "):
            st.write(f"🧑 {message}")
        else:
            st.write(f"🤖 {message}")

def main():
    # Initialize session state
    initialize_session_state()
    
    # App header
    st.title("🤖 Emotion-Based AI Chatbot")
    st.caption("Creator: Abhijith P")
    st.write("This app detects your emotion and lets you chat with an AI that responds based on your mood!")
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("### Step 1: Detect Your Emotion")
        emotion = detect_emotion()
        if emotion:
            st.session_state.emotion = emotion
            st.success(f"Detected Emotion: {emotion}")
    
    with col2:
        st.write("### Step 2: Chat with the AI")
        user_message = st.text_input("Type your message here:", key="user_input")
        
        if st.button("Send", key="send_button"):
            if not st.session_state.emotion:
                st.warning("Please take a photo to detect your emotion first!")
            elif not user_message.strip():
                st.warning("Please enter a message!")
            else:
                with st.spinner("Generating response..."):
                    ai_response = generate_ai_response(user_message, st.session_state.emotion)
                    st.session_state.chat_history.append(f"You: {user_message}")
                    st.session_state.chat_history.append(f"AI: {ai_response}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.divider()
        display_chat_history()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please refresh the page and try again.")
