import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from mlflow.models import infer_signature
from urllib.parse import urlparse
import dagshub

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = sns.load_dataset('titanic')

train_data, test_data = train_test_split(df, test_size=0.25, random_state=42)

X_train, X_test, y_train, y_test = (
    train_data.drop('survived', axis=1),
    test_data.drop('survived', axis=1),
    train_data['survived'],
    test_data['survived']
)

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaling', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder())
])

numerical_columns = X_train.select_dtypes('number').columns
categorical_columns = X_train.select_dtypes('object').columns

preprocessor = ColumnTransformer([
    ('numerical', numerical_pipeline, numerical_columns),
    ('categorical', categorical_pipeline, categorical_columns)
])

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# params = {
#     'solver' : 'lbfgs', 
#     'penalty' : 'l2', 
#     'C' : 0.1, 
#     'max_iter' : 1000, 
#     'class_weight' : 'balanced'
# }

params = {
    'solver': 'lbfgs',         # Algorithm to use in optimization
    'penalty': 'l2',           # Regularization type ('l1', 'l2', 'elasticnet', 'none')
    'C': 1.0,                  # Inverse of regularization strength; smaller values specify stronger regularization
    'max_iter': 100,           # Maximum number of iterations taken for the solvers to converge
    'class_weight': None,      # Weights associated with classes in the form {class_label: weight} or 'balanced'
    'random_state': 42,        # Random state for reproducibility
    'multi_class': 'auto',     # Strategy for multi-class classification ('auto', 'ovr', 'multinomial')
    'verbose': 0,              # For verbosity in the solver
    'tol': 1e-4,               # Tolerance for stopping criteria
    'n_jobs': None,            # Number of CPUs used for parallel computation (if solver supports it)
    'l1_ratio': None           # Elastic-net mixing parameter (only if `penalty='elasticnet'`)
}

def evaluate(actual, pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1_sc = f1_score(y_test, y_pred)

    return accuracy, precision, recall, f1_sc

mlflow.set_experiment('Titanic Model')
# mlflow.set_tracking_uri('http://127.0.0.1:5000/')

# For DAGSHUB
dagshub.init(repo_owner='anonymous298', repo_name='mlflow-practice', mlflow=True)

with mlflow.start_run():

    mlflow.set_tags({
    "dataset": "Titanic",
    "model": "LogisticRegression",
    "type": "Classification"
})

    model = LogisticRegression()
    model.set_params(**params)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy, precision, recall, f1_score = evaluate(y_test, y_pred)

    mlflow.log_params(params)

    mlflow.log_metric('accuracy score', accuracy)
    mlflow.log_metric('precision', precision)
    mlflow.log_metric('recall', recall)
    mlflow.log_metric('f1 score', f1_score)

    predictions = model.predict(X_train)
    signature = infer_signature(X_train, predictions)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    print(mlflow.get_tracking_uri())

    if tracking_url_type_store != 'file':

        mlflow.sklearn.log_model(
            model, 'model', registered_model_name='TitanicLogisticModel', signature=signature
        )

    else:
        mlflow.sklearn.log_model(model, 'model', signature=signature)


