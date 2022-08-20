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

@app.route('/resultats_tags.html', methods=['GET', 'POST'])
def resultats_tags():
    title=request.form.get['title']
    body=request.form.get['body']
    title_cleaned = cleaned_text(title, 'title')
    body_cleaned = cleaned_text(body, 'body')
    doc = list(set(title_cleaned)) + " " + list(set(body_cleaned))
    y_pred = lr_model.predict_proba(doc)
    tags_prediction = inv_transform(y_pred)
    return render_template("result.html", body=body, title=title, prediction=tags_prediction)

if __name__ == "__main__":
    app.run(debug=True)