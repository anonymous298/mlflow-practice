import mlflow
import pandas as pd
import numpy as np

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

X, y = load_wine(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = RandomForestClassifier(n_estimators=150)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print(pred)

def evaluate(actual, pred):
    accuracy = accuracy_score(actual, pred)
    # precision = precision_score(actual, pred)
    # recall = recall_score(actual, pred)

    return accuracy

accuracy = evaluate(y_test, pred)

with mlflow.start_run():
    mlflow.log_param('n_estimators', 150)

    mlflow.log_metric('accuracy', accuracy)