import kfp.dsl as dsl
from kfp.components import load_component_from_file

train_op = load_component_from_file('train_component.yaml')
evaluate_op = load_component_from_file('evaluate_component.yaml')

@dsl.pipeline(
    name='SVM Pipeline',
    description='Trains SVM model on Iris dataset'
)
def svm_pipeline():
    train_task = train_op()
    evaluate_task = evaluate_op(train_task.outputs['model_path'])

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(svm_pipeline, 'svm_pipeline.yaml')

