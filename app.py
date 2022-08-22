import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
from functions import *


app = Flask(__name__)

path = "C:/Users/moumouni/Desktop/OC_P5_ml/"
lr_model = joblib.load(path + "final_model.pkl")
@app.route('/')
def formulaire():
    return render_template("formula.html")

@app.route('/result.html', methods=['GET', 'POST'])
def resultats_tags():
    title=request.form['title']
    body=request.form['body']
    title_cleaned = sentence_cleaner(title)
    body_cleaned = sentence_cleaner(body)
    doc = title_cleaned  + body_cleaned
    y_pred = lr_model.predict_proba(np.array([doc,]))
    tags_pred = inv_transform(y_pred)
    return render_template("result.html", body=body, title=title, tags_prediction=tags_pred)

if __name__ == "__main__":
    app.run(debug=True)