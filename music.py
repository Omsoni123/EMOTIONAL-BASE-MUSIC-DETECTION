import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser

# Load model and data
model = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Define EmotionProcessor class for video processing
class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)

            pred = label[np.argmax(model.predict(lst))]
            cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

            np.save("emotion.npy", np.array([pred]))

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# Set default session state
if "run" not in st.session_state:
    st.session_state["run"] = "true"

# Load emotion data if available
try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

# App title and description
st.set_page_config(
    page_title="Emotion-Based Song Recommendation",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="expanded",  # Changed sidebar state to expanded for better visibility
)

st.title("Emotion-Based Song Recommendation")
st.markdown("---")
st.write("Welcome to the Emotion-Based Song Recommendation App!")

# Background image and styling (Replace 'project1.jpg' with your actual image file path)
background_image = "project1.jpg"
background_style = f"""
    <style>
        .reportview-container {{
            background-image: url('{background_image}');
            background-size: cover;
        }}
    </style>
"""
st.markdown(background_style, unsafe_allow_html=True)

# Main content with red background color (Removed the red background color since the image is now used as background)
st.write("Enter the language and singer's name to get song recommendations based on your emotion.")

# Instructions for users
st.write("Instructions:")
st.write("- Allow camera access to detect your emotion.")
st.write("- Enter the language and singer's name in the fields below.")
st.write("- Click 'Recommend me song' to get song recommendations.")

# Input fields
lang = st.text_input("Language")
singer = st.text_input("Singer")

# WebRTC streamer for video processing
if lang and singer:
    webrtc_streamer(key="key", desired_playing_state=True, video_processor_factory=EmotionProcessor)

# Button for song recommendation
btn = st.button("Recommend me song")

# Song recommendation logic
if btn:
    if not emotion:
        st.warning("Please allow camera access to detect your emotion.")
        st.session_state["run"] = "true"
    else:
        webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}")
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = "false"

# Feedback section
st.write("")
st.write("If you have any feedback or suggestions, please feel free to share.")
feedback = st.text_area("Your feedback")
if st.button("Submit Feedback"):
    with open("feedback.txt", "a") as f:
        f.write(feedback + "\n")
    st.write("Thank you for your feedback! We appreciate it.")
