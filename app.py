# app.py
import streamlit as st
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

@st.cache_resource
def load_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')

load_nltk()

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('spam_classifier.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# Text preprocessing function (must be identical to training)
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    cleaned = [ps.stem(token) for token in tokens if token.isalnum() and token not in stop_words]
    return " ".join(cleaned)

# Streamlit UI
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📱")
st.title("📱 SMS Spam Classifier")
st.markdown("Enter an SMS message below and the model will predict whether it is **Spam** or **Ham** (not spam).")

# User input
user_input = st.text_area("Your message:", height=150)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Preprocess
        processed = transform_text(user_input)
        # Vectorize
        vec = vectorizer.transform([processed]).toarray()
        # Predict
        prediction = model.predict(vec)[0]
        result = "🚨 SPAM" if prediction == 1 else "✅ HAM"
        if prediction == 1:
            st.error(result)
        else:
            st.success(result)