from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import preprocessing as prep
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import ImportData


def get_attributes(dataframe):
    num_columns = ImportData.num_columns(dataframe)-1
    return list(dataframe.columns.values)[:num_columns]

def get_label(dataframe):
    num_columns = ImportData.num_columns(dataframe)-1
    return list(dataframe.columns.values)[num_columns]

def build_decision_tree(attributes, label, max_depth=None):
    return DecisionTreeClassifier(max_depth=max_depth).fit(attributes, label)

def label_encode(dataframe):
    return dataframe.apply(prep.LabelEncoder().fit_transform)

def get_tree_depth(tree): 
    return tree.get_depth

def main():
    training_df = label_encode(ImportData.get_training_set())
    testing_df = label_encode(ImportData.get_test_set()) 

    

    training_feature_list = get_attributes(training_df)
    training_feature_cols = training_df[training_feature_list]
    training_label = training_df.label

    train_training_features, prediction_training_features, train_training_label, prediction_training_label  = \
        train_test_split(training_feature_cols, training_label, test_size=0.2, random_state=1) 

    # test_data_cols = testing_df[training_feature_list]
    

    for tree_depth in range(2, 11): 
        decisiontree = build_decision_tree(train_training_features, train_training_label, tree_depth)
        prediction = decisiontree.predict(prediction_training_features)
        accuracy = metrics.accuracy_score(prediction_training_label, prediction)
        print("Accuracy of Decision Tree with Depth ", tree_depth, ": ", accuracy*100)

    # tree with default depth
    decisiontree = build_decision_tree(train_training_features, train_training_label)
    prediction = decisiontree.predict(prediction_training_features)
    accuracy = metrics.accuracy_score(prediction_training_label, prediction)
    print("Accuracy of Decision Tree with Depth ", decisiontree.get_depth(), ": ", accuracy*100)

    # use entire training dataset to build tree before testing accuracy with testing set
    

    

if __name__ == "__main__":
    main()


# #split dataset in features and target variable
# feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
# X = pima[feature_cols] # Features
# y = pima.label # Target variable
# # Split dataset into training set and test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test