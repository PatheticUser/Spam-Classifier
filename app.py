import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK resources safely
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

# Initialize Porter Stemmer
ps = PorterStemmer()


def transform_text(text):
    """Preprocess text by cleaning and standardizing"""
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    cleaned_tokens = [
        ps.stem(token)
        for token in tokens
        if token.isalnum()
        and token not in stopwords.words("english")
        and token not in string.punctuation
    ]

    return " ".join(cleaned_tokens)


# Page Configuration
st.set_page_config(page_title="Spam Classifier", page_icon=":detective:", layout="wide")

# Custom CSS with high contrast and clear visibility
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
        color: #000000;
        font-family: Arial, sans-serif;
    }
    .main-container {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 20px;
        max-width: 800px;
        margin: 20px auto;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTextArea textarea {
        background-color: white;
        color: black;
        border: 2px solid #333;
        border-radius: 5px;
        font-size: 16px;
        min-height: 200px;
    }
    .stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        font-weight: bold;
    }
    .result-box {
        padding: 15px;
        border-radius: 5px;
        margin-top: 15px;
        font-size: 18px;
        text-align: center;
    }
    .spam-result {
        background-color: #ffdddd;
        border: 2px solid #ff0000;
        color: #ff0000;
    }
    .not-spam-result {
        background-color: #ddffdd;
        border: 2px solid #00aa00;
        color: #00aa00;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Cached model loading
@st.cache_resource
def load_model():
    try:
        tfidf = pickle.load(open("vectorizer1.pkl", "rb"))
        model = pickle.load(open("model1.pkl", "rb"))
        return tfidf, model
    except FileNotFoundError:
        st.error("Model files not found. Please check your files.")
        return None, None


def main():
    st.title("SMS/Email Spam Classifier")

    # Load model
    tfidf, model = load_model()

    if tfidf and model:
        # Message input
        input_sms = st.text_area(
            "Enter your message:",
            placeholder="Type or paste your SMS/Email here...",
            height=250,
        )

        # Prediction Button
        if st.button("Check for Spam", type="primary"):
            if input_sms.strip():
                with st.spinner("Analyzing message..."):
                    # Preprocess and predict
                    transformed_sms = transform_text(input_sms)
                    vector_input = tfidf.transform([transformed_sms])
                    result = model.predict(vector_input)[0]

                    # Result Display
                    if result == 1:
                        st.markdown(
                            '<div class="result-box spam-result">'
                            "SPAM DETECTED! This message appears to be spam.</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div class="result-box not-spam-result">'
                            "NOT SPAM. This message seems legitimate.</div>",
                            unsafe_allow_html=True,
                        )
            else:
                st.warning("Please enter a message to analyze.")


if __name__ == "__main__":
    main()
