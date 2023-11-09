from flask import Flask, request, render_template, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# Predefined list of problematic keywords
keywords = ["shared", "commercial", "advertiser", "transfer personal information",
           "third parties", "shared", "share", "affliate", "disclose", "third party", "permanently stored", "card information",
           "payment information", "financial", "group company", "cookies", "beacon", "affiliates", "advertising partners",
           "partner","card number", "thirdparty", "overseas", "transfer information", "personal information", "transfer",
           "partners","share","disclosure", "credit card","tracking", "other countries","third-party","prior consent",
           "affiliated companies", "transferred", "accessed", "marketing", "subsidiaries","parent companies", "parent company", "sharing",
           "share your information", "share your personal information", "outside the country", "without giving notice", "sell",
           "business partners", "payment processors", "other organisations", "companies", "payment providers", "their privacy policies",
           "related bodies corporate", "cvv", "payment service providers", "service providers"] 


# Load the TF-IDF vectorizer that was saved during model training
with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Define a function to clean the input text
def preprocess_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    return cleaned_text

# Define a function to highlight problematic phrases within the text
def highlight_problematic(text):
    print(f"Input Text: {text}")
    words = text.split()
    highlighted_text = ""
    i = 0
    while i < len(words):
        matched = False
        for phrase in keywords:
            print(f"Matching phrase: {phrase}")
            phrase_words = phrase.split()
            if i + len(phrase_words) <= len(words):
                if all(phrase_words[j].lower() in words[i + j].lower() for j in range(len(phrase_words))):
                    print(f"Matched phrase: {phrase}")
                    highlighted_text += ' '.join([f'<span class="highlight">{word}</span>' for word in words[i:i + len(phrase_words)]]) + ' '
                    i += len(phrase_words)
                    matched = True
                    break
        if not matched:
            highlighted_text += words[i] + ' '
            i += 1
    print(f"Highlighted Text: {highlighted_text.strip()}")
    return highlighted_text.strip()




@app.route('/', methods=['GET', 'POST'])
def predict():
    highlighted_text = None

    if request.method == 'POST':
        try:
            data = request.form['text']

            # Preprocess the input text
            text = preprocess_text(data)

            # Highlight problematic keywords in the entered text
            highlighted_text = highlight_problematic(text)

            # Vectorize the input text using the fitted TF-IDF vectorizer
            text_tfidf = tfidf_vectorizer.transform([text])

            return render_template('index.html', highlighted_text=highlighted_text)
        except Exception as e:
            return jsonify({'error': str(e)})

    else:
        return render_template('index.html', highlighted_text=highlighted_text)

if __name__ == '__main__':
    app.run(debug=True)