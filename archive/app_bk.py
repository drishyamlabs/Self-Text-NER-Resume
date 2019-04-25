import flask
import pickle
import pandas as pd

import spacy
import json
import random
import logging
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from sklearn.metrics import accuracy_score
import itertools

#from flask import Flask, render_template, request
from werkzeug import secure_filename

# Use pickle to load in the pre-trained model. 
#with open(f'model/bike_model_xgboost.pkl', 'rb') as f:

nlp1 = spacy.load('my_model')
	

app = flask.Flask(__name__, template_folder='templates')

with open('sample.txt', 'r',  encoding="utf8") as file:
    text = file.read()
	
#test the model and evaluate it
doc_to_test=nlp1(text)
d={}

for ent in doc_to_test.ents:
    d[ent.label_]=[]
for ent in doc_to_test.ents:
    d[ent.label_].append(ent.text)
for k, v in d.items():
    v.sort()
    d[k] = [item for item, _ in itertools.groupby(v)]
	
print(d)

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        temperature = flask.request.form['temperature']
        humidity = flask.request.form['humidity']
        windspeed = flask.request.form['windspeed']
        return flask.render_template('main.html',
                                     original_input={'Temperature':temperature,
                                                     'Humidity':humidity,
                                                     'Windspeed':windspeed},
                                     result=d['Name'][0],
                                     )
if __name__ == '__main__':
    app.run()