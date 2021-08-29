from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    pred = ''
    jd = ['']
    jd_type = ['DA','DS','MLE','DE']
    if request.method == 'POST':
        jd = [request.form['jd']]
        idx = model.predict(jd)[0]
        pred = jd_type[idx]
    return render_template('index.html', pred=pred, jd_text=jd[0])


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)