from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import preprocessing as prep
import pandas as pd
import numpy as np
import ImportData


def get_attributes(dataframe):
    num_columns = ImportData.num_columns(dataframe)-1
    return list(dataframe.columns.values)[:num_columns]

def get_label(dataframe):
    num_columns = ImportData.num_columns(dataframe)-1
    return list(dataframe.columns.values)[num_columns]

def build_decision_tree(attributes, label):
    return DecisionTreeClassifier().fit(attributes, label)


def main():
    training_df = ImportData.get_training_set()
    testing_df = ImportData.get_testing_set()
    training_df = training_df.apply(prep.LabelEncoder().fit_transform)
    testing_df = testing_df.apply(prep.LabelEncoder().fit_transform)

    attributes_list = get_attributes(training_df)
    attribute_cols = training_df[attributes_list]
    test_data_cols = testing_df[attributes_list]
    label = training_df.label

    decisiontree = build_decision_tree(attribute_cols, label)
    prediction = decisiontree.predict(test_data_cols)

    accuracy = metrics.accuracy_score(testing_df.label, prediction)
    print(accuracy)
    

    

if __name__ == "__main__":
    main()