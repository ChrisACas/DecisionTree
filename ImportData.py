import numpy as np
import pandas as pd

def import_col_names(filename):
    with open(filename) as file:
        col_names = file.readlines()
        return [col_name.split(':', 1)[0] for col_name in col_names]
    
def export_dataframe(dataframe, filename="export.csv"):
    dataframe.head().to_csv(filename)

def import_file(filename, col_names):
    dataframe = pd.read_csv(filename, names=col_names)
    return dataframe

def print_dataframe(dataframe):
    print(dataframe.head())

def num_instances(dataframe):
    return len(dataframe.index)

def num_columns(dataframe):
    return len(dataframe.columns)

def get_training_set():
    training_filename = "census-income.data"
    col_names_file = "census-income-col_names.txt"
    col_names = import_col_names(col_names_file)
    training_dataframe = import_file(training_filename, col_names)
    return training_dataframe.drop(columns='instance weight')

def get_test_set():
    test_filename = "census-income.test"
    col_names_file = "census-income-col_names.txt"
    col_names = import_col_names(col_names_file)
    testing_dataframe = import_file(test_filename, col_names)
    return testing_dataframe.drop(columns='instance weight')

def main():
    training_filename = "census-income.data"
    test_filename = "census-income.test"
    col_names_file = "census-income-col_names.txt"
    
    col_names = import_col_names(col_names_file)
    
    training_dataframe = import_file(training_filename, col_names)
    print("Training Rows: ", num_instances(training_dataframe))
    print("Training Columns: ", num_columns(training_dataframe))

    testing_dataframe = import_file(test_filename, col_names)
    print("Training Rows: ", num_instances(testing_dataframe))
    print("Training Columns: ", num_columns(testing_dataframe))
    

if __name__ == "__main__":
    main()