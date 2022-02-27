#     DECISION TREE ALGORITHM
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pydotplus import graph_from_dot_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import tree
import graphviz

# Accessing the Dataset csv file using Pandas
dataset = pd.read_csv('Iris.csv')
print(dataset.head())

print(dataset.info())

print(dataset.describe())

print(dataset["Species"].value_counts())

# Editing the dataset inorder to remove the unnecessary columns
le = LabelEncoder()
dataset['Species'] = le.fit_transform(dataset['Species'])
print(dataset.head())
print(dataset[50:55])
print(dataset.tail())

dataset = dataset.drop(['Id'], axis = 1)
print(dataset.head())
print(dataset[50:55])
print(dataset.tail())

# Splitting the dataset into train and test
X = dataset.drop(columns=['Species'])
Y = dataset['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

# decision tree
model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
model.fit(x_train, y_train)
plt.figure(figsize=(10,5))
tree.plot_tree(model,fontsize=8)
plt.show()
'''DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')'''

# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test) * 100)

#To show the final Decision Tree
a = tree.plot_tree(model, feature_names=['SepalLengthCm','SepalWidthCm',' PetalLengthCm',' PetalWidthCm'], class_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree for Iris Classification")
plt.show()

