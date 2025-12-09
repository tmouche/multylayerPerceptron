
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.network import Network
from typing import List

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
        if args[i][0] == "--testing":
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
    
    try:
        df_train = np.array(pd.read_csv(train_file))
    except:
        print(f"Error log: can not process {train_file}")
    try:
        df_test = np.array(pd.read_csv(test_file))
    except:
        print(f"Error log: can not process {test_file}")
    data_train = []
    for i in range(len(df_train)):
        data_train.append({"label":np.array, "data":np.array})
        data_train[-1]["label"] = [1] if df_train[i, 0] == 'M' else [0]
        data_train[-1]["data"] = np.array(df_train[i][1:])
    data_test = []
    for i in range(len(df_test)):
        data_test.append({"label":str, "data":np.array})
        data_test[-1]["label"] = [1] if df_train[i, 0] == 'M' else [0]
        data_test[-1]["data"] = np.array(df_test[i][1:]) 
    try:
        myNet = Network()
        myNet.
    except Exception as e:
        print(f"[FATAL] -> The network's configuration failed: {e}")
        exit(1)
    return


if __name__ == "__main__":
    main()
