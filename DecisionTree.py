from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree
from sklearn import preprocessing as prep
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import ImportData
import matplotlib.pyplot as plt

def display_tree(dtree, attribute_names, label):
    fig = plt.figure(figsize=(25,20), dpi=600)
    tree.plot_tree(dtree, 
                   feature_names=attribute_names,  
                   class_names=label,
                   filled=True)
    fig.savefig("decision_tree.png")
    

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

def main():
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
    for tree_depth in range(2, 13): 
        decisiontree = build_decision_tree(training_feature_cols, training_label_col, tree_depth)
        prediction = decisiontree.predict(training_feature_cols)
        accuracy = metrics.accuracy_score(training_label_col, prediction)
        print("Accuracy of Decision Tree built with training set having depth ", tree_depth, ": ", accuracy*100)

    # Training Set accuracy only. Sci-kit learn determined dpeth
    decisiontree = build_decision_tree(training_feature_cols, training_label_col)
    prediction = decisiontree.predict(training_feature_cols)
    accuracy = metrics.accuracy_score(training_label_col, prediction)
    print("Accuracy of Decision Tree built with training set having depth ", decisiontree.get_depth(), ": ", accuracy*100)

    # Testing Set prediction accuracy with skleanr specific depth
    decisiontree = build_decision_tree(training_feature_cols, training_label_col)
    prediction = decisiontree.predict(testing_feature_cols)
    accuracy = metrics.accuracy_score(testing_label_col, prediction)
    print("\nDecision Tree with Sci-kit Learns determination of best fit depth")
    print("Accuracy of classification with Test Data having depth ", decisiontree.get_depth(), ": ", accuracy*100)

    print("\nOptimal Depth has been found to be 10")

    # Testing Dataset prediction accuracy with found best depth
    decisiontree = build_decision_tree(training_feature_cols, training_label_col, 10)
    prediction = decisiontree.predict(testing_feature_cols)
    accuracy = metrics.accuracy_score(testing_label_col, prediction)
    print("\nDecision Tree with predetermined best depth")
    print("Accuracy of classification with Test Data having depth of ", decisiontree.get_depth(), ": ", accuracy*100)


    display_tree(decisiontree,feature_list, 'label')
    


if __name__ == "__main__":
    main()
