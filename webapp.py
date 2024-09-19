import streamlit as st
import pickle
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
print(stopwords.words('english'))  # This should print the list of English stopwords
from nltk.stem.porter import PorterStemmer

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
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Add custom CSS for background
st.markdown(
    """
    <style>
    body {
        background-image: url('https://images.rawpixel.com/image_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvbHIvcm0zODAtMDcta255Z2FwNmouanBn.jpg'); 
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    .stApp {
        background-color: rgba(50, 50, 40, 0.6);
    }
    
     h1, h2, h3{
        color: white; /* Title color */
    }
    
      .stTextArea>label {
        color: silver; /* Label color */
    }
    
    
    </style>
    """,
    unsafe_allow_html=True
)

st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the Message")

if st.button('Predict'):

    # preprocessing
    transformed_sms = transform_text(input_sms)

    #vectorize
    vector_input = tfidf.transform([transformed_sms])

    # predict
    result = model.predict(vector_input)[0]

    # display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")



