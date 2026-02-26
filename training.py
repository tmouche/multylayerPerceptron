from core.layer import Layer
from core.model import Model
from core.network import Network
from ml_tools.fire import Fire
from ml_tools.optimizers import Optimizer, Nesterov_Accelerated_Gradient, Gradient_Descent, RMS_Propagation, ADAM
from ml_tools.activations import Sigmoid
from ml_tools.initialisations import he_normal


from typing import List, Dict, Sequence
from utils.constant import COLUMNS, DROP_COLUMNS
from utils.logger import Logger
from utils.types import ArrayF, FloatT

import numpy
import pandas
import plotly.graph_objects as go 
import sys

logger = Logger()

def	normalize(
        df:pandas.DataFrame,
        starting_column:int
    ) -> pandas.DataFrame:
    """
    Norm a pandas dataFrame by columns.

    Args:
        df (pandas.DataFrame): The DataFrame who contains the data.
        starting_column (int): The first column who is going to be normed.

    Returns:
        pandas.DataFrame: a normed DataFrame by columns.
    """
    for i in range(starting_column, len(df.columns)):
        act_column = df.iloc[:,i] 
        c_min = min(act_column)
        c_max = max(act_column)
        for x in range(len(act_column)):
            df.iloc[x,i] = (df.iloc[x,i] - c_min) / (c_max - c_min)
    return df

def create_normalized_data(
        training_path:str,
        testing_path:str
    ) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    try:
        df_train = pandas.read_csv(training_path, header=None)
        df_train.columns = COLUMNS
    except:
        print(f"Error log: can not process {training_path}")
        raise FileNotFoundError(training_path)
    try:
        df_test = pandas.read_csv(testing_path, header=None)
        df_test.columns = COLUMNS
    except:
        print(f"Error log: can not process {testing_path}")
        raise FileNotFoundError(testing_path)

    df_train.drop(columns=DROP_COLUMNS, inplace=True)
    df_test.drop(columns=DROP_COLUMNS, inplace=True)
    normalized_df_train:pandas.DataFrame = normalize(df_train, 1)
    normalized_df_test:pandas.DataFrame = normalize(df_test, 1) 

    return normalized_df_train, normalized_df_test

def process_df_1_output(
        df_train:pandas.DataFrame,
        df_test:pandas.DataFrame
    ) -> tuple[List[Dict[str, ArrayF]], List[Dict[str, ArrayF]]]:
    """
    Convert training and testing DataFrames into structured datasets
    for a single binary output model.

    Each row of the input DataFrames is transformed into a dictionary
    containing:
        - "label": Binary target encoded as [1] if the first column is 'M',
                   otherwise [0].
        - "data": Feature vector extracted from remaining columns and
                  converted into a NumPy array.

    Args:
        df_train (pandas.DataFrame):
            Training dataset where the first column contains class labels
            and the remaining columns contain features.

        df_test (pandas.DataFrame):
            Testing dataset with the same structure as `df_train`.

    Returns:
        tuple[
            List[Dict[str, ArrayF]],
            List[Dict[str, ArrayF]]
        ]:
            A tuple containing:
                - Processed training dataset.
                - Processed testing dataset.
    """
    data_train: List[Dict[str, ArrayF]] = list()
    for i in range(len(df_train)):
        data_train.append(dict())
        data_train[-1]["label"] = numpy.array([1] if df_train.iloc[i, 0] == 'M' else [0], dtype=FloatT)
        data_train[-1]["data"] = numpy.array(df_train.iloc[i, 1:], dtype=FloatT)

    data_test: List[Dict[str, ArrayF]] = list()
    for i in range(len(df_test)):
        data_test.append(dict())
        data_test[-1]["label"] = numpy.array([1] if df_test.iloc[i, 0] == 'M' else [0], dtype=FloatT)
        data_test[-1]["data"] = numpy.array(df_test.iloc[i, 1:], dtype=FloatT)
    return data_train, data_test

def process_df_2_output(
        df_train:pandas.DataFrame,
        df_test:pandas.DataFrame
    ) -> tuple[List[Dict[str, ArrayF]], List[Dict[str, ArrayF]]]:
    """
    Convert training and testing DataFrames into structured datasets
    for a two-output (one-hot encoded) classification model.

    Each row of the input DataFrames is transformed into a dictionary
    containing:
        - "label": One-hot encoded target vector:
            - [1, 0] if the first column is 'M'
            - [0, 1] otherwise
        - "data": Feature vector extracted from remaining columns and
                  converted into a NumPy array.

    Args:
        df_train (pandas.DataFrame):
            Training dataset where the first column contains class labels
            and the remaining columns contain features.

        df_test (pandas.DataFrame):
            Testing dataset with the same structure as `df_train`.

    Returns:
        tuple[
            List[Dict[str, ArrayF]],
            List[Dict[str, ArrayF]]
        ]:
            A tuple containing:
                - Processed training dataset.
                - Processed testing dataset.
    """
    data_train: List[Dict[str, ArrayF]] = list()
    for i in range(len(df_train)):
        data_train.append({})
        data_train[-1]["label"] = numpy.array([1, 0], dtype=FloatT) if df_train.iloc[i, 0] == 'M' else numpy.array([0, 1], dtype=FloatT)
        data_train[-1]["data"] = numpy.array(df_train.iloc[i, 1:], dtype=FloatT)

    data_test: List[Dict[str, ArrayF]] = list()
    for i in range(len(df_test)):
        data_test.append({})
        data_test[-1]["label"] = numpy.array([1, 0], dtype=FloatT) if df_test.iloc[i, 0] == 'M' else numpy.array([0, 1], dtype=FloatT)
        data_test[-1]["data"] = numpy.array(df_test.iloc[i, 1:], dtype=FloatT)
    return data_train, data_test


def main():

    if len(sys.argv) != 3:
        logger.error("python training.py *path_to_training_dataset* *path_to_testing_dataset*")
        return 1

    train_file: str = sys.argv[1]
    test_file: str = sys.argv[2]
    
    df_train, df_test = create_normalized_data(training_path=train_file, testing_path=test_file)
    l_train, l_test = process_df_2_output(df_train=df_train, df_test=df_test)

    try:
        model = Model()

        model.create_network([
                Layer(shape=9),
                Layer(shape=16, activation="Sigmoid", initializer=he_normal),
                Layer(shape=2, activation="Sigmoid", initializer=he_normal)
            ],
            0.0025,
            2
        )

        opti: Optimizer = ADAM(model.fire, model.network, momentum_rate=0.8, velocity_rate=0.8)

        model.fit(
            optimizer=opti.stochastic,
            ds_train=l_train,
            ds_test=l_test,
            loss="mean_square_error",
            epochs=1000,
            early_stoper=0.003,
            print_training_state=True
        )
    except Exception as exc:
        if exc: logger.error(exc)
        return 1

    epoch = [i for i in range(len(model.accuracies.get("training")))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epoch, y=model.accuracies.get("testing"), name="accuracies testing", line={'color': 'darkred', 'width': 4}))
    fig.add_trace(go.Scatter(x=epoch, y=model.accuracies.get("training"), name="accuracies training", line={'color': 'firebrick', 'width': 4}))
    fig.add_trace(go.Scatter(x=epoch, y=model.losses.get("testing"), name="losses testing", line={'color': 'darkslateblue', 'width': 4}))
    fig.add_trace(go.Scatter(x=epoch, y=model.losses.get("training"), name="losses training", line={'color': 'dodgerblue', 'width': 4}))

    fig.update_layout(
        title=dict(
            text="Accuracies and losses throught the epochs"
        )
    )

    fig.write_html("plots/training_recap.html", auto_open=True)
    return 0



if __name__ == "__main__":
    exit(main())
