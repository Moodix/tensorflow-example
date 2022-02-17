import uuid

import numpy as np
import tensorflow as tf
import valohai

import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from numpy import where


def log_metadata(epoch, logs):
    """Helper function to log training metrics"""
    with valohai.logger() as logger:
        logger.log('epoch', epoch)
        logger.log('accuracy', logs['accuracy'])
        logger.log('loss', logs['loss'])


def main():
    # valohai.prepare enables us to update the valohai.yaml configuration file with
    # the Valohai command-line client by running `valohai yaml step train_model.py`

    valohai.prepare(
        step='train-model',
        default_inputs={
            'dataset': 'http://www.testifytech.ml/Traffic_train.csv',
            'datatest': 'http://www.testifytech.ml/Traffic_test3.csv',

        },
    )

    # Read input files from Valohai inputs directory
    # This enables Valohai to version your training data
    # and cache the data for quick experimentation


    # Print metrics out as JSON
    # This enables Valohai to version your metadata
    # and for you to use it to compare experiments
    input_path = valohai.inputs('dataset').path()
    test_path = valohai.inputs('datatest').path()

    data = pd.read_csv(input_path)
    one_hot_encoded_data = pd.get_dummies(data, columns = ['Code'])
    df = one_hot_encoded_data[["Delay", "Code_200", "Code_201", "Code_204", "Code_302", "Code_400", "Code_404", "Code_500"]]
    model = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
    model.fit(df)

    # Evaluate the model and print out the test metrics as JSON
    testset = pd.read_csv(test_path)
    one_hot_encoded_data2 = pd.get_dummies(testset, columns = ['Code'])
    df2 = one_hot_encoded_data2[["Delay", "Code_200", "Code_201", "Code_204", "Code_302", "Code_400", "Code_404", "Code_500","Y"]]
    X_test=df2[["Delay", "Code_200", "Code_201", "Code_204", "Code_302", "Code_400", "Code_404", "Code_500"]]
    y_test=df2[["Y"]]
    y_pred = model.predict(X_test)

    with valohai.logger() as logger:
        logger.log('y_pred', y_pred)

    # Write output files to Valohai outputs directory
    # This enables Valohai to version your data
    # and upload output it to the default data store

    suffix = uuid.uuid4()
    output_path = valohai.outputs().path(f'model-{suffix}.h5')
    model.save(output_path)


if __name__ == '__main__':
    main()
