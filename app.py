from datetime import date
from pickle import GET
from flask import Flask
from flask import request
from flask.json import jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import json
import numpy as np
from sklearn.metrics import roc_curve,confusion_matrix,accuracy_score,precision_score,precision_score,recall_score,f1_score



# print("1", request)
# print("2", request.form['date'])
# print("3", data)
df = pd.read_csv('data/diabetes.csv')

X = df[['Age', 'Dry Lean Mass', 'Body Fat Mass', 'Time',
        'lost weight', 'Initial Weight', 'Total Body Water', 'Gender','illness']]
Y = df[['Trainer']]

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.1, random_state=0)

gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)

score=gbc.score(X_test, y_test)
print("accuracy of gradient boast classifier",round(score*100,2))

a = gbc.predict([[23, 85, 29, 97, 60, 80, 58, 1,5]])
print(a)

# 
app = Flask(__name__)
# 
# 
def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()
# 
# 
@app.route("/run", methods=['POST'])
def hello_world():
    req = request.get_json()
    valuse = req['array']
    print(valuse)
    df = pd.read_csv('data/diabetes.csv')
# 
    X = df[['Age', 'Dry Lean Mass', 'Body Fat Mass', 'Time',
            'lost weight', 'Initial Weight', 'Total Body Water', 'Gender','illness']]
    Y = df[['Trainer']]
# 
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.1, random_state=0)
# 
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    a = gbc.predict([valuse])
    print(a[0])
# 
    return json.dumps({"resp": a[0]}, default=np_encoder)
# 
# 
if __name__ == '__main__':
    app.run()
