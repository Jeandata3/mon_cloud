import cv2
import streamlit as st
from PIL import Image
from io import BytesIO
import matplotlib as plt
import numpy as np
import cv2 
import time
st.sidebar.title("Parametres")
path = 'haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(path)

#a = st.sidebar.slider("scaleFactor", 0.0, 1.0, 0.1, 0.01)
#b = st.sidebar.slider(" minNeighbors", 5, 20, 5, 5)
#st.sidebar.info(
    #"La tolérance est le seuil de la reconnaissance faciale. Plus la tolérance est faible, plus la reconnaissance faciale est stricte. Plus la tolérance est élevée, plus la reconnaissance faciale est faible..")
#COULEUR = st.sidebar.color_picker("Couleur du rectangle",'#F90034')


# Dictionnaire pour stocker les noms et les images des individus connus
known_faces = {}

def recognize_face(face):
    # Effectuer la reconnaissance du visage
    # Ici, vous pouvez utiliser votre modèle mis à jour pour reconnaître le visage

    # Placeholder pour l'exemple
    return "individu"

def detect():
    rects = face_detector.detectMultiScale(gray_s, 
        scaleFactor=1.1,
        minNeighbors=5, 
        minSize=(30, 30), 
        flags=cv.CASCADE_SCALE_IMAGE)

    for rect in rects:
        cv.rectangle(gray_s, rect, 255, 2)

cap = cv2.VideoCapture(0)
t0 = time.time()

M = np.float32([[0.5, 0, 0], [0, 0.5, 0]])
size = (640, 360)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_s = cv2.warpAffine(gray, M, size)

    detect()
    
    cv2.imshow('window', gray_s)
    t = time.time()
    t0 = t

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Demander le nom de l'individu
name = st.text_input("Nom de l'individu")

    # Enregistrer l'image capturée avec le nom de l'individu dans le dictionnaire
known_faces[name] = frame

def capture_and_label():
    # Initialiser la webcam
    cap = cv2.VideoCapture(0)
    while True:
        # Lire les images de la webcam
        ret, frame = cap.read()
        # Afficher les images
        cv2.imshow('Capture et labélisation', frame)
        # Quitter la boucle lorsque la touche 'q' est pressée
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Arrêter la capture vidéo
    cap.release()
    cv2.destroyAllWindows()

    # Demander le nom de l'individu
    name = st.text_input("Nom de l'individu")

    # Enregistrer l'image capturée avec le nom de l'individu dans le dictionnaire
    known_faces[name] = frame

cap.release()
cv.destroyAllWindows()

def app():
    st.title('PLATEFORME DE DETECTION DE VISAGE')

    st.header("Bienvenue sur notre plateforme de detection de visage !")
    # Ajouter un bouton pour détecter les visages
    if st.button("Détecter les visages"):
        # Appeler la fonction detect_faces
        detect()

    st.write("Appuyez sur le bouton ci-dessous pour capturer et labéliser un nouvel individu")
    # Ajouter un bouton pour capturer et labéliser un nouvel individu
    if st.button("Capturer et labéliser"):
        # Appeler la fonction capture_and_label
        capture_and_label()


if __name__ == "__main__":
    app()




