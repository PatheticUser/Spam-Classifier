import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure required NLTK data is downloaded
nltk.download("punkt")
nltk.download("stopwords")

# Initialize PorterStemmer
ps = PorterStemmer()

# Set page configuration
st.set_page_config(page_title="Spam Classifier", layout="centered")

# --- Custom CSS for Minimalist Design ---
st.markdown(
    """
    <style>
        body {
            background-color: #f5f5f5;
            color: #333;
            font-family: 'Helvetica Neue', sans-serif;
        }
        .stButton > button {
            background-color: #007bff;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 20px;
            border: none;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #0056b3;
        }
        .stTextArea textarea {
            border-radius: 8px;
            border: 1px solid #ddd;
            padding: 12px;
            font-size: 16px;
        }
        .stApp {
            max-width: 600px;
            margin: auto;
            padding: 20px;
        }
        h1 {
            text-align: center;
            color: #007bff;
            font-size: 2.5rem;
            margin-bottom: 20px;
        }
        .result-box {
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            text-align: center;
            font-size: 1.2rem;
            font-weight: bold;
        }
        .spam {
            background-color: #ffebee;
            color: #c62828;
            border: 1px solid #c62828;
        }
        .not-spam {
            background-color: #e8f5e9;
            color: #2e7d32;
            border: 1px solid #2e7d32;
        }
        .warning {
            background-color: #fff3e0;
            color: #ef6c00;
            border: 1px solid #ef6c00;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Text Transformation Function ---
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize
    text = [i for i in text if i.isalnum()]  # Remove non-alphanumeric characters
    text = [
        i
        for i in text
        if i not in stopwords.words("english") and i not in string.punctuation
    ]  # Remove stopwords and punctuations
    text = [ps.stem(i) for i in text]  # Apply stemming
    return " ".join(text)


# --- Load Vectorizer and Model ---
tfidf = pickle.load(open("vectorizer1.pkl", "rb"))
model = pickle.load(open("model1.pkl", "rb"))

# --- Main App Layout ---
st.markdown("<h1>Spam Classifier</h1>", unsafe_allow_html=True)

# Initialize session state for input_sms
if "input_sms" not in st.session_state:
    st.session_state.input_sms = ""

# Input field for SMS/Email message
input_sms = st.text_area(
    "Enter the message:",
    value=st.session_state.input_sms,
    placeholder="Type your email or SMS here...",
    key="input_sms",
    height=150,
)

# Buttons for Clear and Predict
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Clear Message", key="clear_btn"):
        st.session_state.input_sms = ""
        st.rerun()

with col2:
    if st.button("Predict", key="predict_btn"):
        if input_sms.strip():
            with st.spinner("Classifying message..."):
                # Preprocess the input message
                transformed_sms = transform_text(input_sms)

                # Vectorize the input message
                vector_input = tfidf.transform([transformed_sms])

                # Predict whether it's spam or not
                result = model.predict(vector_input)[0]

                # Display the result with styled box
                if result == 1:
                    st.markdown(
                        '<div class="result-box spam">This message is **Spam**!</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="result-box not-spam">This message is **Not Spam**!</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown(
                '<div class="result-box warning">Please enter a valid message for classification.</div>',
                unsafe_allow_html=True,
            )
