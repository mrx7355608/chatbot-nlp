import random
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import string
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

lemmatizer = WordNetLemmatizer() # Initialize a lammetizer
stop_words = set(stopwords.words("english")) # Get a list of stop words

# Convert POS tag to WordNet format so that the wordnet's
# lammetizer can understand them
def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def preprocess(text):
    text = text.lower() # Normalize by converting to lowercase
    text = text.translate(str.maketrans("", "", string.punctuation)) # Remove punctuation
    tokens = word_tokenize(text) # Tokenize
    pos_tags = pos_tag(tokens) # POS tagging

    # Lammetization
    filtered_tokens = [
        lemmatizer.lemmatize(w, get_wordnet_pos(t))
        for w, t in pos_tags if w not in stop_words
    ]
    return " ".join(filtered_tokens)


# Load the file
with open("intents.json", "r") as file:
    data = json.load(file)

texts = []
labels = []

# Extract and preprocess each pattern to make a mapping between
# potential user inputs and their related intents
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        cleaned = preprocess(pattern)
        if not cleaned.strip():
            continue

        texts.append(cleaned.strip())
        labels.append(intent["tag"])



# TF-IDF implementation
# 1. Convert the parsed text in texts array into unigrams and bigrams
# 2. Then calculate their scores using TF-IDF formulas and make a vector
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(texts)

# Encode labels into unique integers so that our model
# can understand them (model cannot understand raw text)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Train model
model = MultinomialNB()
model.fit(X, y)

# Predict intent using model
def predict_intent(text):
    cleaned = preprocess(text) # Clean the data using NLP pipeline
    features = vectorizer.transform([cleaned]) # Vectorize with TF-IDF
    predicted_label = model.predict(features)[0]

    # Converts the intent label integer back to its original form "text"
    intent = label_encoder.inverse_transform([predicted_label])[0]
    return intent


while True:
    # Main flow
    user_input = str(input("You (type 'exit' to quit): "))
    if user_input == "exit":
        break

    intent = predict_intent(user_input)
    responses_map = {
        intent["tag"]: intent["responses"]
        for intent in data["intents"]
    }

    response = random.choice(responses_map.get(intent, ["I don't understand."]))
    print(f"🤖 BOT: {response}\n")

