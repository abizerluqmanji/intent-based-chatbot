from flask import Flask, render_template, request, jsonify
import random
import pickle
import json

# Load trained model and vectorizer
with open("model.pkl", "rb") as model_file:
    clf = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load intents JSON file
with open("intents.json") as file:
    intents = json.load(file)

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]

    # Transform user input into vectorized form
    user_vector = vectorizer.transform([user_input])

    # Predict intent and calculate confidence score
    probs = clf.predict_proba(user_vector)
    max_prob = max(probs[0])
    predicted_tag = clf.predict(user_vector)[0]

    # Check confidence threshold
    print("Max Probability:", max_prob)
    if max_prob < 0.25:
        return jsonify({"response": "I'm not sure I understand. Can you rephrase?"})

    # Fetch response from intents.json based on predicted tag
    for intent in intents["intents"]:
        if intent["tag"] == predicted_tag:
            response = random.choice(intent["responses"])
            return jsonify({"response": response})

    # Fallback response (shouldn't reach here due to confidence check)
    return jsonify({"response": "I'm sorry, I didn't understand that."})


if __name__ == "__main__":
    app.run(debug=True)
