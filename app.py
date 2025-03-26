import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# NLTK Resource Management
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

# Text Preprocessing
ps = PorterStemmer()


def transform_text(text):
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
st.set_page_config(page_title="SpamShield", layout="wide")

# Dark Theme CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #181818;
        color: #E0E0E0;
    }
    .main-container {
        background-color: #222;
        border-radius: 12px;
        padding: 30px;
        max-width: 700px;
        margin: auto;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    h1 {
        color: #4CAF50;
        text-align: center;
        font-weight: 700;
    }
    .stTextArea textarea {
        background-color: #2C2C2C;
        color: #E0E0E0;
        border: 2px solid #4CAF50;
        border-radius: 8px;
        font-size: 16px;
        padding: 15px;
    }
    .stButton > button {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 8px;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #45a049 !important;
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Cached Model Loading
@st.cache_resource
def load_model():
    try:
        tfidf = pickle.load(open("vectorizer1.pkl", "rb"))
        model = pickle.load(open("model1.pkl", "rb"))
        return tfidf, model
    except FileNotFoundError:
        st.error("Model files not found. Please check your configuration.")
        return None, None


def main():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown("<h1>SpamShield</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:#bbb;'>AI-Powered Spam Detection System</p>",
        unsafe_allow_html=True,
    )

    # Load Model
    tfidf, model = load_model()

    if tfidf and model:
        # Message Input and Controls
        col1, col2 = st.columns([4, 1])

        with col1:
            input_sms = st.text_area(
                "Enter your message:",
                placeholder="Type or paste your SMS or email content here...",
                key="message_input",
                help="Enter a message to check if it is classified as spam.",
            )

        with col2:
            if st.button("Clear", key="clear_btn"):
                st.session_state.message = ""
                st.experimental_rerun()

        # Live Character Counter
        char_count = len(input_sms)
        st.markdown(
            f"<p style='color:#bbb;'>Character count: {char_count}</p>",
            unsafe_allow_html=True,
        )

        # Prediction Button
        if st.button("Analyze Message", type="primary"):
            if input_sms.strip():
                with st.spinner("Analyzing message..."):
                    transformed_sms = transform_text(input_sms)
                    vector_input = tfidf.transform([transformed_sms])
                    result = model.predict(vector_input)[0]

                    if result == 1:
                        st.error(
                            "Spam Detected: This message has been flagged as suspicious."
                        )
                    else:
                        st.success("Not Spam: This message appears to be safe.")
            else:
                st.warning("Please enter a message to analyze.")

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
