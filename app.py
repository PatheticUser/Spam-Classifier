import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK resources (if not already present)
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
    """
    Preprocess text by:
    1. Converting to lowercase
    2. Tokenizing
    3. Removing non-alphanumeric characters
    4. Removing stopwords and punctuation
    5. Applying stemming
    """
    # Lowercase conversion
    text = text.lower()

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Remove non-alphanumeric and filter out stopwords/punctuation
    cleaned_tokens = [
        ps.stem(token)
        for token in tokens
        if token.isalnum()
        and token not in stopwords.words("english")
        and token not in string.punctuation
    ]

    return " ".join(cleaned_tokens)


# Page Configuration
st.set_page_config(page_title="Spam Shield", page_icon=":shield:", layout="centered")

# Custom CSS for modern, clean design
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f4f6f9;
        font-family: 'Inter', sans-serif;
    }
    .main-container {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 30px;
        max-width: 600px;
        margin: 20px auto;
    }
    .stTextArea textarea {
        border-radius: 8px;
        border: 1.5px solid #e0e4e8;
        background-color: #f9fafb;
        padding: 12px;
        font-size: 16px;
    }
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #2563eb;
        transform: translateY(-2px);
    }
    .spam-result {
        background-color: #fee2e2;
        color: #7f1d1d;
        border-left: 4px solid #ef4444;
    }
    .not-spam-result {
        background-color: #d1fae5;
        color: #064e3b;
        border-left: 4px solid #10b981;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load pre-trained model and vectorizer
@st.cache_resource
def load_model():
    try:
        tfidf = pickle.load(open("vectorizer1.pkl", "rb"))
        model = pickle.load(open("model1.pkl", "rb"))
        return tfidf, model
    except FileNotFoundError:
        st.error(
            "Model files not found. Please ensure vectorizer1.pkl and model1.pkl exist."
        )
        return None, None


# Main App
def main():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.title("Spam Shield")
    st.markdown("*Intelligent Message Classification*")

    # Load model
    tfidf, model = load_model()

    if tfidf and model:
        # Message input
        input_sms = st.text_area(
            "Paste your message here",
            height=200,
            placeholder="Enter SMS or email content to check for spam...",
        )

        col1, col2 = st.columns(2)

        with col1:
            # Predict Button
            if st.button("Analyze Message", type="primary"):
                if input_sms.strip():
                    with st.spinner("Processing..."):
                        # Preprocess and predict
                        transformed_sms = transform_text(input_sms)
                        vector_input = tfidf.transform([transformed_sms])
                        result = model.predict(vector_input)[0]

                        # Result display
                        if result == 1:
                            st.markdown(
                                '<div class="spam-result" style="padding: 15px; border-radius: 8px;">'
                                "<strong>Spam Detected!</strong><br>"
                                "This message appears to be spam.</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                '<div class="not-spam-result" style="padding: 15px; border-radius: 8px;">'
                                "<strong>Safe Message</strong><br>"
                                "This message looks legitimate.</div>",
                                unsafe_allow_html=True,
                            )
                else:
                    st.warning("Please enter a message to analyze.")

        with col2:
            # Clear Button
            if st.button("Clear Message"):
                input_sms = ""

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
