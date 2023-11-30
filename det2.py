import cv2
import streamlit as st
from PIL import Image
from io import BytesIO
import IPython.display as ipd
import matplotlib as plt
import face_recognition as frg
import yaml
#from utils import recognize, build_dataset
st.sidebar.title("Parametres")

# Put slide to adjust tolerance
a = st.sidebar.slider("scaleFactor", 0.0, 1.0, 0.1, 0.01)
b = st.sidebar.slider(" minNeighbors", 5, 20, 5, 5)
st.sidebar.info(
    "La tolérance est le seuil de la reconnaissance faciale. Plus la tolérance est faible, plus la reconnaissance faciale est stricte. Plus la tolérance est élevée, plus la reconnaissance faciale est faible..")
COULEUR = st.sidebar.color_picker("Couleur du rectangle",'#F90034')
# Dictionnaire pour stocker les noms et les images des individus connus
known_faces = {}
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default .xml')
def detect_faces():
    # Initialize the webcam
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    while True:
        # Read the frames from the webcam
        frame = cv2.imread(r".\images\cap.jpg")
        # Convert the frames to grayscale
        # image = frame[:, :, ::-1]
        # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR5552GRAY)
        # plt.imshow(gray)
        # Detect the faces using the face cascade classifier
        faces = face_cascade.detectMultiScale(frame, scaleFactor=a, minNeighbors=b)
        # faces = face_cascade.detectMultiScale(frame,scaleFactor=1.3, minNeighbors=5)
        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (COULEUR), 2)
        # Display the frames
        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)
        if 0xFF != ord('q'):
            continue
        # Exit the loop when 'q' is pressed
        break
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
def app():
    st.title("Face Detection using Viola-Jones Algorithm")
    st.write("Press the button below to start detecting faces from your webcam")
    # Add a button to start detecting faces
    if st.button("Detect Faces"):
        # Call the detect_faces function
        detect_faces()
if __name__ == "__main__":
    app()