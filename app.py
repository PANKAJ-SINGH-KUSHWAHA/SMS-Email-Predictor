import streamlit as st
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

ps = PorterStemmer()

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Custom CSS for Background & Centering Elements
st.markdown(
    """
    <style>
    /* Background image */
    .stApp {
        background: url('https://source.unsplash.com/1600x900/?technology,communication') no-repeat center center fixed;
        background-size: cover;
    }

    /* Centering Banner */
    .banner-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 20px;
    }

    .banner-container img {
        width: 60%;
        border-radius: 10px;
    }

    /* Title Styling */
    .title {
        text-align: center;
        color: #6a1b9a;
        font-size: 32px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title & Banner
st.markdown("<div class='banner-container'>", unsafe_allow_html=True)
st.image("spam_banner.jpg")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<h1 class='title'>📩 Pankaj Singh SMS/Email Spam Predictor 🚀</h1>", unsafe_allow_html=True)

# User Input
input_sms = st.text_area("📩 Enter your message below:", height=100)


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]  # Remove non-alphanumeric characters
    y = [i for i in y if
         i not in set(stopwords.words('english')) and i not in string.punctuation]  # Remove stopwords & punctuation
    y = [ps.stem(i) for i in y]  # Perform stemming

    return " ".join(y)


if st.button("🔍 **Click to Predict**", help="Analyze the SMS for spam detection", type="primary"):
    # Preprocess
    transformed_sms = transform_text(input_sms)

    # Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # Predict
    result = model.predict(vector_input)[0]
    spam_prob = model.predict_proba(vector_input)[0][1]  # Get probability of spam

    # Display Result
    if result == 1:
        st.markdown("<h2 style='color: red; text-align:center;'>🚨 This is SPAM! 🚨</h2>", unsafe_allow_html=True)
        st.error(f"⚠️ Spam Probability: {spam_prob * 100:.2f}%")
    else:
        st.markdown("<h2 style='color: green; text-align:center;'>✅ This is NOT Spam ✅</h2>", unsafe_allow_html=True)
        st.success(f"✔️ Safe Message! Probability: {spam_prob * 100:.2f}%")

    # Show Confidence Level with Progress Bar
    st.progress(spam_prob)
    st.write(f"Spam Probability: {spam_prob * 100:.2f}%")
