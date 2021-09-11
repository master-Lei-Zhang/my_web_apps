from flask import Flask, jsonify, request, render_template
import json
import numpy as np

app = Flask(__name__)

kw = dict()
with open("analytics.txt") as f:
    kw["analytic"] = f.read().strip().split('\n')
with open("data_sci.txt") as f:
    kw["modeling"] = f.read().strip().split('\n')
with open("data_eng.txt") as f:
    kw["data eng"] = f.read().strip().split('\n')
with open("productionization.txt") as f:
    kw["mle"] = f.read().strip().split('\n')

def label_data(text, kw=kw):
    text = text.lower()
    words = [[],[],[],[]]
    cats = ["analytic", "modeling", "mle", "data eng"]
    kai_label_order = {'analytic':0, 'modeling':1, 'mle':2, 'data eng':3}
    for i, cat in enumerate(cats):
        for k in kw[cat][::-1]:
            if words[i]:
                if k in words[i][-1]:
                    continue
            if k in text:
                words[i].append(k)
    total = 0
    for i in range(len(words)):
        total += len(words[i])
        
    res = dict()
    max_cnt = 0
    max_label = None
    final_kw = None
    if total != 0:
        for i, cat in enumerate(cats):
            res[cat] = len(words[i]) / total
            if res[cat] > max_cnt:
                max_cnt = res[cat]
                max_label = cat
                final_kw = words[i]
            res[cat] = ['{0:.2f}'.format(len(words[i]) / total)] + words[i]
    # res['max_label'] = kai_label_order[max_label]
    res['max_label'] = max_label
    return res

@app.route('/', methods=['GET', 'POST'])
def index():
    pred = {'analytic':'', 'modeling':'', 'mle':'', 'data eng':'', 'max_label':''}
    if request.method == 'POST':
        text = request.form['jd']
        pred = label_data(text, kw)
    return render_template('index.html', DA=pred['analytic'], DS=pred['modeling'],
                            MLE=pred['mle'], DE=pred['data eng'], label=pred['max_label'])


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)