import streamlit as st
from deepface import DeepFace
import groq 
import time
import cv2

groq_api_key = "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"
client = groq.Client(api_key=groq_api_key)

def detect_emotion():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            return emotion
        except Exception as e:
            st.error(f"Error detecting emotion: {e}")
            return None
    cap.release()
    return None

def generate_ai_response(user_message, emotion):
    prompt = f"The user is feeling {emotion}. They said: '{user_message}'. Respond appropriately and empathetically."
    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "You are a helpful and empathetic AI."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

def main():
    st.title("Emotion-Based AI Chatbot\nCreator:Abhijith P")
    st.write("This app detects your emotion and lets you chat with an AI that responds based on your mood!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.write("### Step 1: Detect Your Emotion")
    if st.button("Detect Emotion"):
        emotion = detect_emotion()
        if emotion:
            st.session_state.emotion = emotion
            st.success(f"Detected Emotion: {emotion}")
        else:
            st.error("Could not detect emotion. Please try again.")

    st.write("### Step 2: Chat with the AI")
    user_message = st.text_input("Type your message here:")

    if st.button("Send"):
        if "emotion" not in st.session_state:
            st.error("Please detect your emotion first!")
        elif user_message.strip() == "":
            st.error("Please enter a message!")
        else:
            ai_response = generate_ai_response(user_message, st.session_state.emotion)
            
            st.session_state.chat_history.append(f"You: {user_message}")
            st.session_state.chat_history.append(f"AI: {ai_response}")

    st.write("### Chat History")
    for message in st.session_state.chat_history:
        st.write(message)

if __name__ == "__main__":
    main()