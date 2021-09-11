import pickle
import numpy as np
import pandas as pd

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
    cats = ["analytic", "modeling", "data eng", "mle"]
    kai_label_order = {'analytic':0, 'modeling':1, 'mle':2, 'data eng':3} # added by lei for easy label retrieve
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
            # res[cat] = len(words[i]) / total
            # print(cat, words[i])
            res[cat] = ['{0:.2f}'.format(len(words[i]) / total)] + words[i]
            if res[cat] > max_cnt:
                max_cnt = res[cat]
                max_label = cat
                final_kw = words[i]
    # print(max_label)
#     print(final_kw)
    res['max_label'] = kai_label_order[max_label] # added by lei for easy label retrieve
    return res

print(label_data('i am a data scientist', kw))