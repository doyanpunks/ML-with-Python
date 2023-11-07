from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import mglearn
import numpy as np

iris_dataset = load_iris()

# split data 
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0, shuffle=False, stratify=None
)

#print("Target names: {}".format(iris_dataset['target_names']))
#print("Feature names: \n{}".format(iris_dataset['feature_names']))
#print("Iris shape: {}".format(X_train.shape))
#print("{}".format(iris_dataset['data'][:5]))
#print("Type of target: {}".format(type(iris_dataset['target'])))
#print("Type: {}".format((iris_dataset['target'])))

# Create dataframe from data in X_train
# label the columns using the strings in iris_dataset_features_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# create scatter matrix from the dataframe, color by y_train
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o',
                        hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

# Categorize data
knn = KNeighborsClassifier(n_neighbors=1)
# Train the model
knn.fit(X_train, y_train)

# Suppose we found an iris in the wild with a sepal length of 5 cm, a sepal width of 2.9 mc, a petal length of 1 cm, and petal width of 0.2 cm
X_new = np.array([[5,2.9,1,0.2]])
#print("X_new.shape: {}".format(X_new.shape))

prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))

# Evaluation
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))