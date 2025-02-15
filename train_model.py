import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load intents dataset
with open("intents.json") as file:
    data = json.load(file)

# Prepare training data
patterns = []
tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])

# Convert text into numerical format using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X, tags)

# Save the trained model and vectorizer to files
with open("model.pkl", "wb") as model_file:
    pickle.dump(clf, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model training complete. Files saved: model.pkl, vectorizer.pkl")
