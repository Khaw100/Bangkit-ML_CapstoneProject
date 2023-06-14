from flask import Flask, request, jsonify
from prediction import model_predict
import json

app = Flask(__name__)

# Function
def readJSON(filename='data.json'):
    with open(filename, 'r') as read_file:
        data = json.load(read_file)
        return data

@app.route('/', methods=['GET', 'POST'])
def index():
    response = {"Hello": "This is Model Prediction API for Sentiment"}
    return jsonify(response)

@app.route('/predict_int',methods=['GET','POST'])
def predict_int():
    data = request.json['data']
    positif = 0
    negatif = 0
    neutral = 0
    for d in data["details"]:
        result = model_predict(d["comment"])

        if (result == "Positive"):
            positif+=1
        elif (result == "Negative"):
            negatif +=1
        else:
            neutral += 1

    return jsonify({"Positif": positif,
                    "Negatif" : negatif,
                    "Neutral" : neutral})

@app.route('/predict_sentiment',methods=['GET','POST'])
def predict_sentiment():
    result = []
    data = request.json['data']
    for d in data["details"]:
        temp = model_predict(d["comment"])
        result.append(temp)
    return jsonify({"Result": result})

