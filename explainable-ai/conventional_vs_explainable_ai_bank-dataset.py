# Conventional AI vs. Explainable AI - Scenario1 (Iris Dataset)
# Written by Ugur Uresin (https://github.com/ugururesin)
# Repo Github: https://github.com/ugururesin/my_templates/tree/main/explainable-ai


# Conventional AI
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv', delimiter=';')

# Convert categorical features to numeric using one-hot encoding
data = pd.get_dummies(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('y_yes', axis=1), data['y_yes'], test_size=0.2, random_state=42)

# Train a Gradient Boosting classifier
clf = GradientBoostingClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the testing set and calculate accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


# Explaiable AI
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from dtreeviz.trees import dtreeviz

# Load the dataset
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv', delimiter=';')

# Convert categorical features to numeric using one-hot encoding
data = pd.get_dummies(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('y_yes', axis=1), data['y_yes'], test_size=0.2, random_state=42)

# Train a Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Visualize the decision tree
viz = dtreeviz(clf, X_train, y_train, target_name='y_yes', feature_names=list(X_train.columns), class_names=['no', 'yes'])
viz.view()
