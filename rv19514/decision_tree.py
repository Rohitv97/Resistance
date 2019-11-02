import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle

dataset = pd.read_csv("train.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:,10]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy: ", metrics.accuracy_score(y_test,y_pred))

pkl_f = "pickle_dt_model.pkl"
with open(pkl_f, 'wb') as file:
    pickle.dump(clf, file)
