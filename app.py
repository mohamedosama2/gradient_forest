from flask import Flask
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier



app = Flask(__name__)


@app.route("/")
def hello_world():
    df = pd.read_csv('data/diabetes.csv')

    X = df[['Age', 'Dry Lean Mass', 'Body Fat Mass', 'Time',
            'lost weight', 'Initial Weight', 'Total Body Water', 'Gender']]
    Y = df[['Trainer']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.1, random_state=0)

    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    a = gbc.predict([[23, 85, 29, 97, 60, 80, 58, 1]])
    print(a)
    return "<p>hello</p>"
