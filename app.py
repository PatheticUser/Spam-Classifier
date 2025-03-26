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
st.set_page_config(page_title="SpamShield", page_icon=":detective:", layout="wide")

# Advanced Dark Theme CSS
st.markdown(
    """
    <style>
    /* Dark Theme Base */
    .stApp {
        background-color: #121212;
        color: #E0E0E0;
    }

    /* Main Container */
    .main-container {
        background-color: #1E1E1E;
        border-radius: 12px;
        padding: 30px;
        max-width: 700px;
        margin: 20px auto;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        border: 1px solid #333;
    }

    /* Typography */
    h1 {
        color: #4CAF50;
        text-align: center;
        font-weight: 700;
        margin-bottom: 20px;
    }

    /* Text Area Styling */
    .stTextArea textarea {
        background-color: #2C2C2C;
        color: #E0E0E0;
        border: 2px solid #4CAF50;
        border-radius: 8px;
        font-size: 16px;
        padding: 15px;
        min-height: 250px;
    }

    /* Buttons */
    .stButton > button {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #45a049 !important;
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
    }

    /* Result Styling */
    .result-container {
        margin-top: 20px;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-size: 18px;
        font-weight: 600;
    }

    .spam-result {
        background-color: rgba(244, 67, 54, 0.2);
        color: #F44336;
        border: 2px solid #F44336;
    }

    .not-spam-result {
        background-color: rgba(76, 175, 80, 0.2);
        color: #4CAF50;
        border: 2px solid #4CAF50;
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
    # Initialize session state for message
    if "message" not in st.session_state:
        st.session_state.message = ""

    # Main Container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # Title
    st.markdown("<h1>SpamShield</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:#888;'>Intelligent Message Classification</p>",
        unsafe_allow_html=True,
    )

    # Load model
    tfidf, model = load_model()

    if tfidf and model:
        # Message Input and Clear Button
        col1, col2 = st.columns([3, 1])

        with col1:
            # Text Area with session state
            input_sms = st.text_area(
                "Enter your message:",
                value=st.session_state.message,
                placeholder="Type or paste your SMS/Email here...",
                key="message_input",
            )

        with col2:
            # Clear Button
            if st.button("Clear", key="clear_btn"):
                st.session_state.message = ""
                st.experimental_rerun()

        # Prediction Button
        if st.button("Analyze Message", type="primary"):
            if input_sms.strip():
                with st.spinner("Analyzing message..."):
                    # Preprocess and predict
                    transformed_sms = transform_text(input_sms)
                    vector_input = tfidf.transform([transformed_sms])
                    result = model.predict(vector_input)[0]

                    # Result Display
                    result_class = "spam-result" if result == 1 else "not-spam-result"
                    result_text = "SPAM DETECTED!" if result == 1 else "NOT SPAM"

                    st.markdown(
                        f"""
                        <div class="result-container {result_class}">
                            {result_text}
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )
            else:
                st.warning("Please enter a message to analyze.")

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
