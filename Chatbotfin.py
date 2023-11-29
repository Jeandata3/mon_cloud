import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import streamlit as st

st.set_page_config(page_title="🤗ChatBot")

# Chargement et prétraitement des données :
# Load the text file and preprocess the data
with open("pg34648.txt", 'r', encoding='utf-8') as f:
    data = f.read().replace('\n', ' ')
# Tokenize the text into sentences
sentences = sent_tokenize(data)
# Define a function to preprocess each sentence
def preprocess(sentence):
    # Tokenize the sentence into words
    words = word_tokenize(sentence)
    # Remove stopwords and punctuation
    words = [word.lower() for word in words if word.lower() not in stopwords.words('french') and word not in string.punctuation]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

# Preprocess each sentence in the text
corpus = [preprocess(sentence) for sentence in sentences]

# Définition de la fonction de similarité :
# Define a function to find the most relevant sentence given a query
def get_most_relevant_sentence(query):
    # Preprocess the query
    query = preprocess(query)
    # Compute the similarity between the query and each sentence in the text
    max_similarity = 0
    most_relevant_sentence = ""
    for sentence in corpus:
        similarity = len(set(query).intersection(sentence)) / float(len(set(query).union(sentence)))
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_sentence = " ".join(sentence)
    return most_relevant_sentence
# La fonction chatbot :
def chatbot(question):
    # Find the most relevant sentence
    most_relevant_sentence = get_most_relevant_sentence(question)
    # Return the answer
    return most_relevant_sentence

# Create a Streamlit app
def main():
    st.title("ASSISTANT VIRTUEL")
    st.write("Hello 👨🏼‍🦰! je suis un Assistant de lecture virtuel. Merci de me poser des questions sur le livre : les TERRES D'OR.")
    st.subheader(" OEUVRES DE GUSTAVE AIMARD ")
    st.subheader(" Je suis pret!!! ")
    # Get the user's question

    question = st.text_input("Moi:")
    # Create a button to submit the question
    if st.button("soumettre ma question"):
        # Call the chatbot function with the question and display the response
        response = chatbot(question)
        st.write("👨🏼‍voici ma reponse: " + response)
if __name__ == "__main__":
    main()

st.markdown("📖Ceci est un chatbot d'un projet d'etude basé sur un Project Gutenberg eBook of Les terres d'or" )