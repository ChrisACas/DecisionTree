import numpy as np
import pandas as pd

def import_file(filename):
    dataframe = pd.read_csv(filename)
    return dataframe

def print_dataframe(dataframe):
    print(dataframe.head())

def num_instances(dataframe):
    return len(dataframe.index)

def num_columns(dataframe):
    return len(dataframe.columns)

def main():
    filename = "census-income.data"
    dataframe = import_file(filename)

    print("Rows: ", num_instances(dataframe))
    print("Columns: ", num_columns(dataframe))
    print_dataframe(dataframe)

if __name__ == "__main__":
    main()