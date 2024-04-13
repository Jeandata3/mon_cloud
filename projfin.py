import streamlit as st
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
import lazypredict
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pickle
import time
#st.set_page_config(page_title="Application de detection du diabète")
st.set_page_config(page_title='Application de detection du diabète', page_icon=':smiley',
                   layout="wide", initial_sidebar_state='expanded')
def main():
    st.title("APPLICATION DE DETECTION DE DIABETE")
   # st.image("pexels-alex")
    st.image('pexels-michelangelo-buonarroti-8728382.jpg',width=800, use_column_width=300,caption='La technologie au service de la santé')
    #st.image('pexels-nataliya-vaitkevich-6940866.jpg', caption='La technologie au service de la santé')
    st.subheader("Cette application a été conçue dans le but de vous aider à predire la probabilité de contraction du diabète. Veuillez l'exploiter aisement afin de pouvoir ameliorer vos efforts pour un meilleur suivi des personnes suscpetibles de contrcater cette pathologie.")
    st.warning ("Ceci ne représente pas un dispositif médical et ne remplace pas l'avis et les examens médicaux")
    #st.info("📖Instructions : Veuillez activer l'onglet des paramètres en haut à gauche de l'écran en cliquant sur '>' puis renseigner les champs suivant la description indiquée. Après verification des informations renseignées veuillez clicker sur analyse.")
    #st.sidebar.title("INFORMATIONS DU PATIENT")
    # fonction d'importation des données
    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv('diabetes2.csv')
        return data
    # Affichage de la table de données
    df = load_data()
    df_sample = df.sample(100)
    # Creation des variables
    Variables = st.sidebar.write(
        "PARAMETRES DE PREDICTION",
    )
    # paramètres de prediction
    Pregnancies = st.number_input("Pregnancies : le nombre de grossesses contracté par cette femme.",
            0,100, step=1)
    Glucose = st.number_input("Glucose: Taux de glucose dans le plasma sanguin. ", 0.0, 1000.1, 0.1, 0.01)
    BloodPressure = st.number_input("BloodPressure: Pression sanguine diastolique.", 0.0, 1000.1, 0.1, 0.01)
    SkinThickness = st.number_input("SkinThickness : Epaisseur mesurée d’un pli de peau au niveau du triceps en mm. Il s’agit d’une mesure permettant d’estimer l’obésité, ou en tous cas, la couche de graisse sous-cutanée à ce niveau", 0.0, 1000.1, 0.1, 0.01)
    Insulin = st.number_input("Insulin : Détermine la quantité d’insuline deux heures après prise orale de sucre dans un test standardisé, en µU/mL", 0.0, 1000.1, 0.1, 0.01)
    BMI = st.number_input("BMI: IMC- indice de masse corporelle", 0.0, 1000.1, 0.1, 0.01)
    DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction: Indice de prédisposition au diabète établi en fonction des informations sur la famille.", 0.0, 1000.1, 0.1, 0.01)
    Age = st.number_input("Age : Désigne l’age de l’individu", 0,100, step=1)
    loaded_model = pickle.load(open('diabete_trained_modelfin2.sav','rb'))

    def predict(input_data1):
        # prediction avec de nouvelles données
        input_data1 = (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
        # changement des input_data en numpy_array
        input_data1_as_numpy_array = np.asarray(input_data1)
        # reshape the array as we are predicting for one instance
        input_data1_reshaped = input_data1_as_numpy_array.reshape(1, -1)

        prediction = loaded_model.predict(input_data1_reshaped)
        print(prediction)

        if (prediction[0] == 1):
            st.write(
                "📝La probabilité que cette personne developpe le diabète est très élévé. Veuillez confirmer cette analyse par un avis medicale suivie d'une prise en charge.")
            st.date_input("Date d'analyse")


        else:
            st.write(
                "📝La probabilité que cette personne devoloppe le diabète est relative faible. Veuillez toutefois lui conseiller la pratique regulière du sport et une alimentation saine.")
            st.date_input("Date d'analyse")


    resultas = ''
    if st.button("ANALYSE📊"):
        resultats = predict([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(resultas)



if __name__ == '__main__':
    main()
