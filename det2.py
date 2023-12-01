import cv2
import streamlit as st

st.title('PLATEFORME DE DETECTION DE VISAGE')
st.header("🤗😎☺️Bienvenue sur notre plateforme de detection de visage !")
st.sidebar.title("🛠️⚙️Paramètres")
# Put slide to adjust tolerance
scaleFactor = st.sidebar.slider("scaleFactor", 0.0, 1.1 ,3.0, 0.01)
minNeighbors = st.sidebar.slider(" minNeighbors", 1, 3, 10, 1)
Selection_couleur = st.sidebar.color_picker("Couleur du rectangle",'#F90034')
st.sidebar.info(
    "📑La tolérance est le seuil de la reconnaissance faciale. Plus la tolérance est faible, plus la reconnaissance faciale est stricte. Plus la tolérance est élevée, plus la reconnaissance faciale est faible..")


# Charger le classificateur en cascade de Haar pour la détection des visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Dictionnaire pour stocker les noms et les images des individus connus
known_faces = {}
def chang_color(changCol):
    changCol = changCol.lstrip('#')
    return tuple(int(changCol[i:i+2], 16)
for i in (0, 2, 4))

color = chang_color(Selection_couleur)
def detect_faces():
    # Initialiser la webcam
    cap = cv2.VideoCapture(0)
    while True:
        # Lire les images de la webcam
        ret, frame = cap.read()
        # Convertir les images en niveaux de gris
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
               # Détecter les visages en utilisant le classificateur en cascade de visage
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
        # Dessiner des rectangles autour des visages détectés et afficher les noms
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0),color, 2)

            # Effectuer la reconnaissance des visages
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (224, 224))
            label = recognize_face(face_roi)

            # Afficher le nom à côté du rectangle de détection
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Afficher les images
        cv2.imshow('Détection de visage avec l\'algorithme Viola-Jones', frame)
        # Quitter la boucle lorsque la touche 'q' est pressée
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    # Libérer la webcam et fermer toutes les fenêtres
    cap.release()
    cv2.destroyAllWindows()


def recognize_face(face):
    # Effectuer la reconnaissance du visage
    # Ici, vous pouvez utiliser votre modèle mis à jour pour reconnaître le visage

    # Placeholder pour l'exemple
    return "Tresor"


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


def app():
    # Ajouter un bouton pour détecter les visages
    if st.button("Détecter les visages"):
        # Appeler la fonction detect_faces
        detect_faces()

    st.write("Appuyez sur le bouton ci-dessous pour capturer et labéliser un nouvel individu")
    # Ajouter un bouton pour capturer et labéliser un nouvel individu
    if st.button("Capturer et labéliser"):
        # Appeler la fonction capture_and_label
        capture_and_label()


if __name__ == "__main__":
    app()
