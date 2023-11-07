from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import mglearn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(
#     cancer.data, cancer.target, stratify=cancer.target, random_state=66)

# training_accuracy = []
# test_accuracy = []

# # n_neighbors from 1 to 10
# neighbors_settings = range(1,11)

# for n_neighbors in neighbors_settings:
#     # build the model
#     clf = KNeighborsClassifier(n_neighbors=n_neighbors)
#     clf.fit(X_train, y_train)
#     # record training set accuracy
#     training_accuracy.append(clf.score(X_train, y_train))
#     # record generalization accuracy
#     test_accuracy.append(clf.score(X_test, y_test))

# plt.plot(neighbors_settings, training_accuracy, label="traaining accuracy")
# plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("n_neighbors")
# plt.legend()
# mglearn.plots.plot_knn_regression(n_neighbors=1)
# mglearn.plots.plot_knn_regression(n_neighbors=3)
# plt.show()

from sklearn.neighbors import KNeighborsRegressor

X, y = mglearn.datasets.make_wave(n_samples=40)

#split the wave dataset into a training and a test set
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)

# insantiate the model and set the number of neighbors to consider to 3
reg = KNeighborsRegressor(n_neighbors=3)
#fit the model
reg.fit(X_train, y_train)

print("Test set prediction:\n{}".format(reg.predict(X_test)))
