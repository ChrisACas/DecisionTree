from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree
from sklearn import preprocessing as prep
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import ImportData
import matplotlib.pyplot as plt

def display_tree(dtree, attribute_names, label, filename="decision_tree"):
    fig = plt.figure(figsize=(25,20), dpi=600)
    tree.plot_tree(dtree, 
                   feature_names=attribute_names,  
                   class_names=label,
                   filled=True)
    fig.savefig("toy_tree.png")
    

def get_attributes(dataframe):
    num_columns = ImportData.num_columns(dataframe)-1
    return list(dataframe.columns.values)[:num_columns]

def get_label(dataframe):
    num_columns = ImportData.num_columns(dataframe)-1
    return list(dataframe.columns.values)[num_columns]

def build_decision_tree(attributes, label, max_depth=None):
    return DecisionTreeClassifier(criterion='entropy', max_depth=max_depth).fit(attributes, label)

def label_encode(dataframe):
    return dataframe.apply(prep.LabelEncoder().fit_transform)

def get_tree_depth(tree): 
    return tree.get_depth


def census_decision_tree():
    training_df = label_encode(ImportData.get_training_set())
    testing_df = label_encode(ImportData.get_test_set()) 
    
    # trainig dataset
    feature_list = get_attributes(training_df)
    training_feature_cols = training_df[feature_list]
    training_label_col = training_df.label

    # testing dataset
    testing_feature_cols = testing_df[feature_list]
    testing_label_col = testing_df.label
 

    train_training_features, prediction_training_features, train_training_label, prediction_training_label  = \
        train_test_split(training_feature_cols, training_label_col, test_size=0.2, random_state=1) 

    # determine best depth with training set only. 
    print("\nDecision tree built and tested with test/train split of Training Dataset Only")
    for tree_depth in range(2, 11): 
        decisiontree = build_decision_tree(train_training_features, train_training_label, tree_depth)
        prediction = decisiontree.predict(prediction_training_features)
        accuracy = metrics.accuracy_score(prediction_training_label, prediction)
        print("Accuracy of Decision Tree built with training set having depth ", tree_depth, ": ", accuracy*100)

    
    print("\nOptimal Depth has been found to be 9")

    # Testing Dataset prediction accuracy with found best depth
    #   This training set was re-built with the entire data set, no split
    decisiontree = build_decision_tree(train_training_features, train_training_label, 9)
    prediction = decisiontree.predict(testing_feature_cols)
    accuracy = metrics.accuracy_score(testing_label_col, prediction)
    print("\nDecision Tree with predetermined best depth. Model buillt with partial training dataset from earlier testing")
    print("Accuracy of classification with Test Data having depth of ", decisiontree.get_depth(), ": ", accuracy*100)

    # Testing Dataset prediction accuracy with found best depth
    #   This training set was re-built with the entire data set, no split
    decisiontree = build_decision_tree(training_feature_cols, training_label_col, 9)
    prediction = decisiontree.predict(testing_feature_cols)
    accuracy = metrics.accuracy_score(testing_label_col, prediction)
    print("\nDecision Tree with predetermined best depth. Model buillt with entire training dataset")
    print("Accuracy of classification with Test Data having depth of ", decisiontree.get_depth(), ": ", accuracy*100)

    
    display_tree(decisiontree,feature_list, 'label')

def toy_dataset():
    toy_df = ImportData.import_file('toy_dataset.csv')
    
    feature_list = get_attributes(toy_df)
    feature_cols = toy_df[feature_list]
    label_col = toy_df.A

    decisiontree = build_decision_tree(feature_cols, label_col, 1)

    display_tree(decisiontree,feature_list, 'XA')
    

def main():
    # census_decision_tree()
    toy_dataset()    


if __name__ == "__main__":
    main()
