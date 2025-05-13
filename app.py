from flask import Flask, render_template, request, jsonify
import json
import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load intents from the JSON file
with open("intents.json", encoding='utf-8') as file:
    data = json.load(file)

# Prepare the dataset
texts = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern.lower())
        labels.append(intent["tag"])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Vectorize the patterns
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Flask app setup
app = Flask(_name_)

# Predict the intent of the user input
def predict_intent(user_input):
    if not user_input.strip():
        return None
    try:
        input_tfidf = vectorizer.transform([user_input.lower()])
        prediction = model.predict(input_tfidf)[0]
        return prediction
    except:
        return None