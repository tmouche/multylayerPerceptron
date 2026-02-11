from core.network import Network, NetworkConfig
from ml_tools.evaluations import classification
from typing import List, Dict, Sequence
from utils.constant import COLUMNS, DROP_COLUMNS

import numpy
import pandas
import plotly.graph_objects as go 
import sys

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
        raise FileNotFoundError(training_path)

    df_train.drop(columns=DROP_COLUMNS, inplace=True)
    df_test.drop(columns=DROP_COLUMNS, inplace=True)
    normalized_df_train:pandas.DataFrame = normalize(df_train, 1)
    normalized_df_test:pandas.DataFrame = normalize(df_test, 1) 

    return normalized_df_train, normalized_df_test

def process_df_1_output(
        df_train:pandas.DataFrame,
        df_test:pandas.DataFrame
    ) -> tuple[List[Dict[str, numpy.array]], List[Dict[str, numpy.array]]]:
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
            List[Dict[str, numpy.array]],
            List[Dict[str, numpy.array]]
        ]:
            A tuple containing:
                - Processed training dataset.
                - Processed testing dataset.
    """
    data_train: List[Dict[str, numpy.array]] = list()
    for i in range(len(df_train)):
        data_train.append(dict())
        data_train[-1]["label"] = [1] if df_train.iloc[i, 0] == 'M' else [0]
        data_train[-1]["data"] = numpy.array(df_train.iloc[i, 1:])

    data_test: List[Dict[str, numpy.array]] = list()
    for i in range(len(df_test)):
        data_test.append(dict())
        data_test[-1]["label"] = [1] if df_test.iloc[i, 0] == 'M' else [0]
        data_test[-1]["data"] = numpy.array(df_test.iloc[i, 1:])
    return data_train, data_test

def process_df_2_output(
        df_train:pandas.DataFrame,
        df_test:pandas.DataFrame
    ) -> tuple[List[Dict[str, numpy.array]], List[Dict[str, numpy.array]]]:
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
            List[Dict[str, numpy.array]],
            List[Dict[str, numpy.array]]
        ]:
            A tuple containing:
                - Processed training dataset.
                - Processed testing dataset.
    """
    data_train: List[Dict[str, numpy.array]] = list()
    for i in range(len(df_train)):
        data_train.append({})
        data_train[-1]["label"] = numpy.array([1, 0]) if df_train.iloc[i, 0] == 'M' else numpy.array([0, 1])
        data_train[-1]["data"] = numpy.array(df_train.iloc[i, 1:])

    data_test: List[Dict[str, numpy.array]] = list()
    for i in range(len(df_test)):
        data_test.append({})
        data_test[-1]["label"] = numpy.array([1, 0]) if df_test.iloc[i, 0] == 'M' else numpy.array([0, 1])
        data_test[-1]["data"] = numpy.array(df_test.iloc[i, 1:])
    return data_train, data_test


def main():
    argc = len(sys.argv)
    args = []
    i = 0
    while i < argc:
        if sys.argv[i][:2] == "--":
            args.append([])
            args[-1].append(sys.argv[i])
            i+=1
            while i < argc and sys.argv[i][:2] != "--":
                args[-1].append(sys.argv[i])
                i+=1
        elif i > 0:
            print(f"Error log: Unknow token {sys.argv[i]}")
            exit(1)
        else:
            i+=1
    train_file = None
    test_file = None
    for i in range(len(args)):
        if args[i][0] == "--training":
            if len(args[i]) != 2 or train_file:
                print(f"Error log: the training option should be unique and look like %--training *path_to_file*%")
                exit(1)
            train_file = args[i][1]
        elif args[i][0] == "--testing":
            if len(args[i]) != 2 or test_file:
                print(f"Error log: the testing option should be unique and look like %--testing *path_to_file*%")
                exit(1)
            test_file = args[i][1]
        else:
            print(f"Error log: Unknow option {args[i][0]}")
            exit(1)
    if train_file is None or test_file is None:
        print(f"Error log: missing mandatory argument (--init/--training/--testing)")
        exit(1)
    
    df_train, df_test = create_normalized_data(training_path=train_file, testing_path=test_file)
    l_train, l_test = process_df_2_output(df_train=df_train, df_test=df_test)
    EPOCH = 350
    try:
        myNet = Network(NetworkConfig(
            learning_rate=0.0025,
            epoch=EPOCH,
            batch_size=4,
            loss_threshold=0.004,
            shape=[9, 16, 2],
            evaluation=classification,
            activation_name="sigmoid",
            loss_name="mean square error",
            output_activation_name="sigmoid",
            initialisation_name="he uniform",
            optimisation_name="mini_adam"
        ))
        myNet.option_visu_training = True
        accuracies, losses = myNet.train(l_train, l_test)
    except Exception as e:
        print(f"[FATAL] -> The network failed: {e}")
        return
    epoch = [i for i in range(EPOCH)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epoch, y=accuracies["testing"], name="accuracies testing", line={'color': 'darkred', 'width': 4}))
    fig.add_trace(go.Scatter(x=epoch, y=accuracies["training"], name="accuracies training", line={'color': 'firebrick', 'width': 4}))
    fig.add_trace(go.Scatter(x=epoch, y=losses["testing"], name="losses testing", line={'color': 'darkslateblue', 'width': 4}))
    fig.add_trace(go.Scatter(x=epoch, y=losses["training"], name="losses training", line={'color': 'dodgerblue', 'width': 4}))

    fig.update_layout(
        title=dict(
            text="Accuracies and losses throught the epochs"
        )
    )

    fig.write_html("plots/training_recap.html", auto_open=True)



if __name__ == "__main__":
    main()
