import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import median_absolute_error, mean_absolute_error, mean_squared_error, accuracy_score
import pickle
import matplotlib.pyplot as plt

dataset = pd.read_csv("train.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:,10]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)

# fnames = ["spy_prob", "team", "leader", "mission_outcome", "v1", "v2", "v3", "v4", "v5"]
fnames = dataset.columns[1:-1].values.tolist()
tnames = ['spy']

clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy: ", accuracy_score(y_test,y_pred))
print('mean_squared_error: ', mean_squared_error(y_test, y_pred))
print('root_mean_squared_error: ', np.sqrt(mean_squared_error(y_test, y_pred)))
# print('median_absolute_error: ', median_absolute_error(y_test, y_pred))
print('mean_absolute_error: ', mean_absolute_error(y_test, y_pred))


plt.scatter(X_test[:, 0], y_test, color='green', label='Actual')
plt.savefig('DT_scatter_actual_spy-prob-to-spy')
plt.scatter(X_test[:, 0], y_pred, color='blue', label='Predicted')
plt.legend()
plt.savefig('DT_scatter_predicted_spy-prob-to-spy')
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render('dtree')

# dot_data = tree.export_graphviz(clf, out_file=None,
#                                 feature_names=fnames,
#                                 class_names='spy')
#
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_pdf("dtree.pdf")

pkl_f = "pickle_dt_model.pkl"
with open(pkl_f, 'wb') as file:
    pickle.dump(clf, file)
