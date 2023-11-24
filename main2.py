import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import lazypredict
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pickle

def main():
    st.title("APPLICATION DE DETECTION DE CHURN-ENTREPRISE DE TELEPHONIE MOBILE")
    st.subheader("Cette application a été conçue dans le but de vous aider à predire la probabilité de départ de vos clients. Veuillez l'exploiter aisement afin de pouvoir ameliorer vos efforts pour un meilleur suivi de votre base de données clients.")
    st.sidebar.title("OPTIONS")


    # fonction d'importation des données
    #@st.cache_data(persist=True)
    #def load_data():
        #data = pd.read_csv('Expresso_churn_dataset.csv')
        #return data
    # Affichage de la table de données
    #df = load_data()
    #df_sample = df.sample(100)
    # Creation des variables
    #Variables = st.sidebar.write(
        #"PARAMETRES DE PREDICTION",
    #)
    # paramètres de prediction
    Variables = st.sidebar.write("PARAMETRES DE PREDICTION")
    Freq = st.sidebar.number_input("Choisir la fréquence de rechargement mensuelle du client",
            0,100, step=1)
    Anc = st.sidebar.number_input("Choisir l'Anciennété sur le reséau (en mois) du client", 1,100, step=2)
    loaded_model = pickle.load(open('trained_model.sav','rb'))

    def predict(input_data):
        input_data = (Freq, Anc)
        # changement des input_data en numpy_array
        input_data_as_numpy_array = np.asarray(input_data)
        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        prediction = loaded_model.predict(input_data_reshaped)
        print(prediction)

        if (prediction[0] == 0):
           st.write("Selon les paramètres enregistrées, la probabilité que le client reste sur le réseau est relativement elévée, toutefois veuillez renforcer votre stragegie de fidelisation.")
        else:
           st.write("Selon les paramètres enregistrées, la probabilité que le client reste sur le réseau est relativement faible. Il y a un fort risque d'abandon de votre réseau par ce client. Veuillez mettre en place une bonne stratégie de fidélisation.")
    resultas = ''
    if st.button("Execution"):
        resultats = predict([Freq,Anc])
    st.success(resultas)

st.sidebar.write("Copyright (Jean Jacques Gogahi)")





if __name__=='__main__':
    main()
