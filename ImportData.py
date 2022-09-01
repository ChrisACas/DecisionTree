import numpy as np
import pandas as pd

def import_file(filename):
    dataframe = pd.read_csv(filename)
    return dataframe

def print_dataframe(dataframe):
    dataframe.head()


def main():
    filename = "census-income.data"
    dataframe = import_file(filename)

    print_dataframe(dataframe)

if __name__ == "__main__":
    main()