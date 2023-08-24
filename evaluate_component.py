from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from typing import NamedTuple
import kfp

def evaluate_model(model_path: str) -> NamedTuple('Outputs', [('accuracy', float)]):
    model = joblib.load(model_path)
    # Load test data, predict, and evaluate
    # ...

    accuracy = 0.95  # Replace with actual accuracy
    # Create confusion matrix plot and save
    # ...

    return (accuracy,)

if __name__ == '__main__':
    kfp.components.create_component_from_func(evaluate_model, output_component_file='evaluate_component.yaml')


