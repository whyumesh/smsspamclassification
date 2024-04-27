import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import base64

# Download necessary nltk data if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load the pre-trained models
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Set background image using HTML
background_image = """
<style>
body {
    background-image: url('https://cdn.wallpapersafari.com/20/12/A2OCUo.jpg');
    background-size: cover;
}
</style>
"""
st.markdown(background_image, unsafe_allow_html=True)

# Streamlit app title with styling
st.title("📧 Email/SMS Spam Classifier")
st.markdown("---")  # Horizontal line for separation

# Text area for input
input_sms = st.text_area("Enter the message", height=150)  # Increased height

# Button to trigger prediction with styling
if st.button('Predict', key="predict_button"):
    # Preprocess input
    transformed_sms = transform_text(input_sms)
    # Vectorize input
    vector_input = tfidf.transform([transformed_sms])
    # Predict
    result = model.predict(vector_input)[0]
    # Display result with styling
    if result == 1:
        st.error("🚫 Spam")
    else:
        st.success("✅ Not Spam")
    st.markdown("---")  # Horizontal line for separation
    st.write("Would you like to analyze another message?")

# Set page footer
footer = """
---
Built by Umesh Pawar(https://yourwebsite.com)
"""
st.markdown(footer, unsafe_allow_html=True)
