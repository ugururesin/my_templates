# Conventional AI vs. Explainable AI - Scenario1 (Iris Dataset)
# Written by Ugur Uresin
# Github:


# Conventional AI
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print("Accuracy: ", accuracy)


# Explaiable AI
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from dtreeviz.trees import dtreeviz

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print("Accuracy: ", accuracy)

viz = dtreeviz(clf, iris.data, iris.target, target_name='Species',
              feature_names=iris.feature_names, class_names=list(iris.target_names))
viz.view()
