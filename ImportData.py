import numpy as np
import pandas as pd

def import_col_names(filename):
    with open(filename) as file:
        col_names = file.readlines()
        return [col_name.split(':', 1)[0] for col_name in col_names]
    

def import_file(filename, col_names):
    dataframe = pd.read_csv(filename, names=col_names)
    return dataframe

def print_dataframe(dataframe):
    print(dataframe.head())

def num_instances(dataframe):
    return len(dataframe.index)

def num_columns(dataframe):
    return len(dataframe.columns)


def main():
    filename = "census-income.data"
    col_names = "census-income-col_names.txt"
    col_names = import_col_names(col_names)
    

    print(len(col_names))
    print(col_names)
    dataframe = import_file(filename, col_names)

    print("Rows: ", num_instances(dataframe))
    print("Columns: ", num_columns(dataframe))
    dataframe.head().to_csv("export.csv")
    

if __name__ == "__main__":
    main()