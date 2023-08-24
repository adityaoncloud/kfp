import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
from typing import NamedTuple
import kfp

def train_svm() -> NamedTuple('Outputs', [('model_path', str)]):
    data = load_iris()
    X = data.data
    y = data.target
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC()
    model.fit(X_train, y_train)
    model_path = 'model.pkl'
    joblib.dump(model, model_path)
    return (model_path,)

if __name__ == '__main__':
    kfp.components.create_component_from_func(train_svm, output_component_file='train_component.yaml')

