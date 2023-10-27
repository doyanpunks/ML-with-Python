from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import mglearn
iris_dataset = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0
)

#print("Target names: {}".format(iris_dataset['target_names']))
#print("Feature names: \n{}".format(iris_dataset['feature_names']))
#print("{}".format(iris_dataset['data'][:5]))
#print("Type of taret: {}".format(type(iris_dataset['target'])))
#print("Type: {}".format((iris_dataset['target'])))

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o',
                        hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
