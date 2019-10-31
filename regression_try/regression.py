import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

def mregression_train():
    dataset = pd.read_csv("train.csv")
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:,10]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

    from sklearn.preprocessing import StandardScaler

    # sc_X = StandardScaler()
    # X_train = sc_X.fit_transform(X_train)
    # X_test = sc_X.fit_transform(X_test)
    # sc_y = StandardScaler()
    # y_train = sc_y.fit_transform(y_train)

    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    pkl_f = "pickle_mr_model.pkl"
    with open(pkl_f, 'wb') as file:
        pickle.dump(regressor, file)

    # y_pred = regressor.predict(is_spy)
    #
    # plt.scatter(X_test[:, 0], y_test, color='green')
    # plt.savefig('scatter_actual')

def mregression_pred(is_spy):
    pkl_f = "regression_try/pickle_mr_model.pkl"

    with open(pkl_f, 'rb') as file:
        pickle_model = pickle.load(file)

    y_pred = pickle_model.predict(is_spy)
    return y_pred
