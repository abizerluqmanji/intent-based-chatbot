from flask import Flask, render_template, request, jsonify
import random
import pickle

# Load trained model and vectorizer
with open("model.pkl", "rb") as model_file:
    clf = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load intents JSON file
import json

with open("intents.json") as file:
    intents = json.load(file)

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    user_vector = vectorizer.transform([user_input])
    predicted_tag = clf.predict(user_vector)[0]

    for intent in intents["intents"]:
        if intent["tag"] == predicted_tag:
            response = random.choice(intent["responses"])
            return jsonify({"response": response})

    return jsonify({"response": "I'm sorry, I didn't understand that."})


if __name__ == "__main__":
    app.run(debug=True)
