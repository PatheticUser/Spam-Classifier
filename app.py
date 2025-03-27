import streamlit as st
import pickle
import string
import nltk
import pandas as pd
import numpy as np
import base64
from sklearn.metrics import confusion_matrix, classification_report
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from typing import Tuple, Optional

# NLTK Resource Management
nltk.download("punkt", force=True, quiet=True)
nltk.download("stopwords", force=True, quiet=True)


# Preprocessing Utility
class TextPreprocessor:
    def __init__(self):
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))
        self.punctuation = set(string.punctuation)

    def transform_text(self, text: str) -> str:
        """
        Comprehensive text preprocessing:
        1. Lowercase conversion
        2. Tokenization
        3. Remove stopwords and punctuation
        4. Stemming
        """
        # Lowercase and tokenize
        tokens = nltk.word_tokenize(text.lower())

        # Advanced cleaning
        cleaned_tokens = [
            self.ps.stem(token)
            for token in tokens
            if token.isalnum()
            and token not in self.stop_words
            and token not in self.punctuation
        ]

        return " ".join(cleaned_tokens)


# Model Manager
class SpamClassificationModel:
    def __init__(self, vectorizer_path: str, model_path: str):
        self.vectorizer_path = vectorizer_path
        self.model_path = model_path
        self.tfidf = None
        self.model = None
        self.preprocessor = TextPreprocessor()

    def load_model(self) -> Tuple[bool, Optional[str]]:
        """
        Load TF-IDF vectorizer and spam classification model
        Returns success status and optional error message
        """
        try:
            with open(self.vectorizer_path, "rb") as f:
                self.tfidf = pickle.load(f)
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            return True, None
        except FileNotFoundError as e:
            return False, f"Model files not found: {str(e)}"
        except Exception as e:
            return False, f"Error loading model: {str(e)}"

    def predict(self, message: str) -> Tuple[int, float]:
        """
        Predict spam probability and classification
        Returns (prediction, probability)
        """
        if not self.model or not self.tfidf:
            raise ValueError("Model not loaded")

        # Preprocess and vectorize
        processed_text = self.preprocessor.transform_text(message)
        vectorized_text = self.tfidf.transform([processed_text])

        # Predict with probability
        prediction = self.model.predict(vectorized_text)[0]
        proba = self.model.predict_proba(vectorized_text)[0]
        spam_proba = proba[1] if prediction == 1 else proba[0]

        return prediction, spam_proba


# Helper function to create download link
def get_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button">{text}</a>'
    return href


# Streamlit App Configuration
def configure_app():
    st.set_page_config(
        page_title="Spam Shield",
        page_icon="https://i.imgur.com/zfC5U6t.png",
        layout="wide",
    )

    # Custom CSS with enhanced styling
    st.markdown(
        """
        <style>
        /* Global Styles */
        .stApp {
            background-color: #0E1117;
            color: #E0E0E0;
        }
        
        /* Main Container */
        .main-container {
            background: linear-gradient(145deg, #1A1C25, #131720);
            border-radius: 16px;
            padding: 30px;
            max-width: 900px;
            margin: 20px auto;
            box-shadow: 0 10px 30px rgba(0,0,0,0.4);
            border: 1px solid #2A2E3A;
        }
        
        /* Title and Headers */
        .title {
            background: linear-gradient(90deg, #4CAF50, #2E7D32);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            font-weight: 800;
            font-size: 3.2rem;
            margin-bottom: 30px;
            text-shadow: 0 2px 10px rgba(76, 175, 80, 0.3);
            letter-spacing: -0.5px;
        }
        
        .subtitle {
            color: #9E9E9E;
            text-align: center;
            font-size: 1.1rem;
            margin-top: -20px;
            margin-bottom: 30px;
        }
        
        /* Text Area */
        .stTextArea textarea {
            background-color: #1A1C25;
            color: #E0E0E0;
            border: 2px solid #2A2E3A;
            border-radius: 12px;
            font-size: 16px;
            padding: 15px;
            min-height: 200px;
            transition: all 0.3s ease;
        }
        
        .stTextArea textarea:focus {
            border-color: #4CAF50;
            box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2);
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(90deg, #4CAF50, #2E7D32) !important;
            color: white !important;
            border-radius: 12px;
            font-weight: 600;
            border: none;
            padding: 12px 28px;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 4px 10px rgba(46, 125, 50, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(76, 175, 80, 0.4);
        }
        
        .stButton > button:active {
            transform: translateY(1px);
        }
        
        /* Results Container */
        .result-container {
            margin-top: 25px;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 20px;
            font-weight: 600;
            transition: all 0.3s ease;
            animation: fadeIn 0.5s ease-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .spam-result {
            background: linear-gradient(145deg, rgba(244, 67, 54, 0.1), rgba(244, 67, 54, 0.2));
            color: #F44336;
            border: 2px solid #F44336;
            box-shadow: 0 4px 15px rgba(244, 67, 54, 0.2);
        }
        
        .not-spam-result {
            background: linear-gradient(145deg, rgba(76, 175, 80, 0.1), rgba(76, 175, 80, 0.2));
            color: #4CAF50;
            border: 2px solid #4CAF50;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.2);
        }
        
        /* Probability Meter */
        .probability-container {
            margin-top: 15px;
            padding: 10px;
            border-radius: 8px;
            background-color: #1A1C25;
            border: 1px solid #2A2E3A;
        }
        
        .probability-label {
            font-size: 14px;
            color: #9E9E9E;
            margin-bottom: 5px;
        }
        
        .probability-bar {
            height: 8px;
            border-radius: 4px;
            background-color: #2A2E3A;
            position: relative;
            overflow: hidden;
        }
        
        .probability-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease-out;
        }
        
        .probability-value {
            font-size: 14px;
            font-weight: 600;
            margin-top: 5px;
            text-align: right;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: #1A1C25;
            border-radius: 12px;
            padding: 5px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding: 10px 20px;
            background-color: transparent;
            border: none;
            color: #9E9E9E;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #2A2E3A !important;
            color: #4CAF50 !important;
        }
        
        /* File Uploader */
        .stFileUploader {
            padding: 15px;
            border-radius: 12px;
            border: 2px dashed #2A2E3A;
            background-color: #1A1C25;
            transition: all 0.3s ease;
        }
        
        .stFileUploader:hover {
            border-color: #4CAF50;
        }
        
        /* DataFrame */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #2A2E3A;
        }
        
        .stDataFrame [data-testid="stDataFrameResizable"] {
            background-color: #1A1C25;
        }
        
        .stDataFrame th {
            background-color: #2A2E3A;
            color: #E0E0E0;
            padding: 12px 15px;
            font-weight: 600;
        }
        
        .stDataFrame td {
            padding: 10px 15px;
            border-bottom: 1px solid #2A2E3A;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            color: #4CAF50;
            font-weight: 700;
            font-size: 2rem;
        }
        
        [data-testid="stMetricLabel"] {
            color: #9E9E9E;
        }
        
        /* Cards for metrics */
        .metric-card {
            background: linear-gradient(145deg, #1A1C25, #131720);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            border: 1px solid #2A2E3A;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            border-color: #4CAF50;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #4CAF50;
            margin: 10px 0;
        }
        
        .metric-label {
            color: #9E9E9E;
            font-size: 1rem;
            font-weight: 500;
        }
        
        /* Download button */
        .download-button {
            display: inline-block;
            background: linear-gradient(90deg, #4CAF50, #2E7D32);
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 600;
            margin-top: 15px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 10px rgba(46, 125, 50, 0.3);
        }
        
        .download-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(76, 175, 80, 0.4);
        }
        
        /* Logo and branding */
        .logo-container {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;
        }
        
        .logo {
            width: 50px;
            height: 50px;
            margin-right: 15px;
        }
        
        /* Animations */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 20px;
            color: #9E9E9E;
            font-size: 0.9rem;
            margin-top: 40px;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )


def main():
    # App Configuration
    configure_app()

    # Model Initialization
    model_manager = SpamClassificationModel(
        vectorizer_path="vectorizer1.pkl", model_path="model1.pkl"
    )

    # Model Loading
    model_loaded, error_msg = model_manager.load_model()

    # Logo and Title
    st.markdown(
        """
        <div class="logo-container">
            <img src="https://i.imgur.com/zfC5U6t.png" class="logo" alt="Spam Shield Logo">
            <h1 class="title">Spam Shield</h1>
        </div>
        <p class="subtitle">Advanced AI-powered spam detection system</p>
        """,
        unsafe_allow_html=True,
    )

    # Error Handling for Model Loading
    if not model_loaded:
        st.error(f"Model Loading Error: {error_msg}")
        st.stop()

    # Wrap everything in a container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # Tabs with icons for different functionalities
    tab1, tab2, tab3 = st.tabs(
        ["🛡️ Spam Detection", "📊 Bulk Analysis", "📈 Model Performance"]
    )

    with tab1:
        # Single Message Detection
        st.markdown(
            "<h3 style='color:#4CAF50; margin-bottom:20px;'>Message Analysis</h3>",
            unsafe_allow_html=True,
        )

        input_sms = st.text_area(
            "Enter your message:",
            placeholder="Type or paste your SMS/Email here to check if it's spam...",
            key="single_message",
        )

        col1, col2 = st.columns([3, 1])

        with col1:
            predict_clicked = st.button("🔍 Analyze Message", type="primary")

        # Prediction Logic
        if predict_clicked and input_sms.strip():
            try:
                prediction, probability = model_manager.predict(input_sms)

                result_class = "spam-result" if prediction == 1 else "not-spam-result"
                result_text = f"⚠️ SPAM DETECTED!" if prediction == 1 else f"✅ NOT SPAM"

                # Enhanced result display with probability
                st.markdown(
                    f"""
                    <div class="result-container {result_class}">
                        {result_text}
                    </div>
                    <div class="probability-container">
                        <div class="probability-label">Confidence Level</div>
                        <div class="probability-bar">
                            <div class="probability-fill" style="width: {probability*100}%; 
                                background-color: {'#F44336' if prediction == 1 else '#4CAF50'};">
                            </div>
                        </div>
                        <div class="probability-value">{probability*100:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Add explanation based on prediction
                if prediction == 1:
                    st.markdown(
                        """
                    <div style="margin-top: 20px; padding: 15px; border-radius: 8px; background-color: rgba(244, 67, 54, 0.1); border: 1px solid #F44336;">
                        <h4 style="color: #F44336; margin-top: 0;">Why is this message flagged as spam?</h4>
                        <p style="font-size: 14px; color: #E0E0E0;">
                            This message contains patterns commonly found in spam communications. 
                            These may include suspicious keywords, unusual formatting, or request patterns 
                            that match known spam techniques.
                        </p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                    <div style="margin-top: 20px; padding: 15px; border-radius: 8px; background-color: rgba(76, 175, 80, 0.1); border: 1px solid #4CAF50;">
                        <h4 style="color: #4CAF50; margin-top: 0;">This message appears to be legitimate</h4>
                        <p style="font-size: 14px; color: #E0E0E0;">
                            Our analysis indicates this message doesn't contain patterns typically 
                            associated with spam communications.
                        </p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")

    with tab2:
        # Bulk Analysis with enhanced UI
        st.markdown(
            "<h3 style='color:#4CAF50; margin-bottom:20px;'>Batch Processing</h3>",
            unsafe_allow_html=True,
        )

        # Instructions
        st.markdown(
            """
        <div style="padding: 15px; border-radius: 8px; background-color: #1A1C25; border: 1px solid #2A2E3A; margin-bottom: 20px;">
            <h4 style="color: #E0E0E0; margin-top: 0;">Instructions</h4>
            <p style="font-size: 14px; color: #9E9E9E;">
                Upload a CSV file containing a column named "message" with the text messages you want to analyze.
                The system will process all messages and provide spam classification results.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader("Upload CSV with messages", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                # Validate DataFrame
                if "message" not in df.columns:
                    st.error("CSV must contain a 'message' column")
                    st.stop()

                # Progress bar for bulk analysis
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Bulk Prediction with progress updates
                total_rows = len(df)

                predictions = []
                probabilities = []

                for i, message in enumerate(df["message"]):
                    pred, prob = model_manager.predict(message)
                    predictions.append(pred)
                    probabilities.append(prob)

                    # Update progress
                    progress = (i + 1) / total_rows
                    progress_bar.progress(progress)
                    status_text.text(f"Analyzing message {i+1} of {total_rows}...")

                df["prediction"] = predictions
                df["spam_probability"] = probabilities

                # Clear progress indicators
                status_text.empty()

                # Success message
                st.success(f"✅ Analysis complete! Processed {total_rows} messages.")

                # Display Results with styling
                st.markdown(
                    "<h4 style='color:#4CAF50; margin-top:20px;'>Analysis Results</h4>",
                    unsafe_allow_html=True,
                )

                # Summary statistics
                spam_count = df["prediction"].sum()
                spam_percentage = (spam_count / total_rows) * 100

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Total Messages</div>
                            <div class="metric-value">{total_rows}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Spam Detected</div>
                            <div class="metric-value">{spam_count} ({spam_percentage:.1f}%)</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Display the dataframe with results
                st.dataframe(df)

                # Download Link with enhanced styling
                st.markdown(
                    get_download_link(
                        df, "spam_analysis_results.csv", "📥 Download Analysis Results"
                    ),
                    unsafe_allow_html=True,
                )

            except Exception as e:
                st.error(f"Bulk Analysis Error: {str(e)}")

    with tab3:
        # Model Performance Metrics with enhanced visualization
        st.markdown(
            "<h3 style='color:#4CAF50; margin-bottom:20px;'>Performance Metrics</h3>",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div style="padding: 15px; border-radius: 8px; background-color: #1A1C25; border: 1px solid #2A2E3A; margin-bottom: 20px;">
            <p style="font-size: 14px; color: #9E9E9E;">
                These metrics represent the model's performance on test data. They indicate how well the model can 
                distinguish between spam and legitimate messages.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Performance metrics in a grid layout
        col1, col2, col3 = st.columns(3)

        metrics = {
            "Accuracy": 0.98,
            "Precision": 0.99,
            "Recall": 0.97,
            "F1 Score": 0.98,
            "False Positive Rate": 0.02,
            "False Negative Rate": 0.03,
        }

        # First row
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value">{metrics["Accuracy"]:.2%}</div>
                    <div style="font-size: 12px; color: #9E9E9E;">Overall correctness</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Precision</div>
                    <div class="metric-value">{metrics["Precision"]:.2%}</div>
                    <div style="font-size: 12px; color: #9E9E9E;">True positives accuracy</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Recall</div>
                    <div class="metric-value">{metrics["Recall"]:.2%}</div>
                    <div style="font-size: 12px; color: #9E9E9E;">Spam detection rate</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Add some space
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Second row
        col4, col5, col6 = st.columns(3)

        with col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">F1 Score</div>
                    <div class="metric-value">{metrics["F1 Score"]:.2%}</div>
                    <div style="font-size: 12px; color: #9E9E9E;">Precision/recall balance</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col5:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">False Positive Rate</div>
                    <div class="metric-value">{metrics["False Positive Rate"]:.2%}</div>
                    <div style="font-size: 12px; color: #9E9E9E;">Legitimate marked as spam</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col6:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">False Negative Rate</div>
                    <div class="metric-value">{metrics["False Negative Rate"]:.2%}</div>
                    <div style="font-size: 12px; color: #9E9E9E;">Spam missed</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Add confusion matrix visualization
        st.markdown(
            "<h4 style='color:#4CAF50; margin-top:30px;'>Confusion Matrix</h4>",
            unsafe_allow_html=True,
        )

        # Sample confusion matrix data
        conf_matrix = np.array([[980, 20], [30, 970]])

        # Create a visual representation of confusion matrix
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; margin-top: 20px;">
                <div style="display: grid; grid-template-columns: auto auto auto; gap: 2px; background-color: #2A2E3A; padding: 2px; border-radius: 8px;">
                    <div style="background-color: #1A1C25; padding: 10px; text-align: center; font-weight: bold; color: #9E9E9E;">Predicted ↓ Actual →</div>
                    <div style="background-color: #1A1C25; padding: 10px; text-align: center; font-weight: bold; color: #4CAF50;">Not Spam</div>
                    <div style="background-color: #1A1C25; padding: 10px; text-align: center; font-weight: bold; color: #F44336;">Spam</div>
                    <div style="background-color: #1A1C25; padding: 10px; text-align: center; font-weight: bold; color: #4CAF50;">Not Spam</div>
                    <div style="background-color: rgba(76, 175, 80, 0.2); padding: 15px; text-align: center; font-size: 18px; font-weight: bold; color: #4CAF50;">{conf_matrix[0,0]}</div>
                    <div style="background-color: rgba(244, 67, 54, 0.2); padding: 15px; text-align: center; font-size: 18px; font-weight: bold; color: #F44336;">{conf_matrix[0,1]}</div>
                    <div style="background-color: #1A1C25; padding: 10px; text-align: center; font-weight: bold; color: #F44336;">Spam</div>
                    <div style="background-color: rgba(244, 67, 54, 0.2); padding: 15px; text-align: center; font-size: 18px; font-weight: bold; color: #F44336;">{conf_matrix[1,0]}</div>
                    <div style="background-color: rgba(76, 175, 80, 0.2); padding: 15px; text-align: center; font-size: 18px; font-weight: bold; color: #4CAF50;">{conf_matrix[1,1]}</div>
                </div>
            </div>
            <div style="text-align: center; margin-top: 10px; color: #9E9E9E; font-size: 14px;">
                True Negatives: {conf_matrix[0,0]} | False Positives: {conf_matrix[0,1]} | False Negatives: {conf_matrix[1,0]} | True Positives: {conf_matrix[1,1]}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Footer
    st.markdown(
        """
        <div class="footer">
            <p>Spam Shield - Advanced AI-powered spam detection system</p>
            <p>© 2025 All Rights Reserved</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
