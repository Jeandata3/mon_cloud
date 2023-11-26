import streamlit as st
import pandas as pd
import numpy as np
import pickle

def main():
    st.title("APPLICATION DE PREDICTION-OUVERTURE DE COMPTE BANCAIRE")
    st.subheader("Cette application a été conçue dans le but de vous aider à identifier quelles sont les personnes les plus susceptibles d'avoir ou d'utiliser un compte bancaire ")
    st.sidebar.title("OPTIONS")


    # fonction d'importation des données
    #@st.cache_data(persist=True)
    #def load_data():
        #data = pd.read_csv('Financial_inclusion_dataset.csv')
        #return data
    # Affichage de la table de données
    #df = load_data()
    #df_sample = df.sample(100)
    # Creation des variables
    Variables = st.sidebar.write(
        "PARAMETRES DE PREDICTION",
    )

    location_type = st.sidebar.selectbox("location_type : 'Rural-0 / Urban-1' ",
                                         (1,0))
    cellphone_access = st.sidebar.selectbox("cellphone_access : 'Yes-1 / No-0' ",
                                            (1,0))
    household_size = st.sidebar.number_input("household_size",
            0,100, step=1)
    age_of_respondent = st.sidebar.number_input("age_of_respondent",
            0,100, step=1)
    gender_of_respondent = st.sidebar.selectbox("gender_of_respondent : 'Male-1 / Female-0'",
                                                (1,0))
    relationship_with_head = st.sidebar.selectbox("relationship_with_head : 'Spouse-5','Head of Household-1','Other relative-3','Child-0','Parent-4'",
                                                  (5,1,3,0,4))
    marital_status = st.sidebar.selectbox("marital_status : 'Married/Living together-2','Widowed-4','Single/Never Married-0','Divorced/Seperated-3'",
                                          (2,4,0,3))
    education_level = st.sidebar.selectbox("education_level:'Secondary education-3,'No formal education-0','Vocational/Specialised training-5','Primary education-2'",
                                           (3,0,5,2))
    job_type = st.sidebar.selectbox("job_type : 'Self employed-9','Government Dependent-4','Formally employed Private-3','Informally employed-5','No Income-6'",
                                    (9,4,4,3,5,6))
    loaded_model = pickle.load(open('def_inclusion_trained_model.sav', 'rb'))

    def predict(input_data1):
        input_data1 = (location_type, cellphone_access,household_size,age_of_respondent,gender_of_respondent,relationship_with_head,marital_status,education_level,job_type)
        # changement des input_data en numpy_array
        input_data1_as_numpy_array = np.asarray(input_data1)
        # reshape the array as we are predicting for one instance
        input_data1_reshaped = input_data1_as_numpy_array.reshape(1, -1)

        prediction = loaded_model.predict(input_data1_reshaped)
        print(prediction)

        if (prediction[0] == 1):
            st.write(
                "la personne est susceptible d'ouvrir un compte bancaire")
        else:
            st.write(
                "la personne n'est pas susceptible d'ouvrir un compte bancaire")

    resultas = ''
    if st.button("Prediction"):
        resultats = predict([location_type, cellphone_access,household_size,age_of_respondent,gender_of_respondent,relationship_with_head,marital_status,education_level,job_type])
    st.success(resultas)

if __name__=='__main__':
    main()
