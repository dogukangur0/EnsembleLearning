from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt


dataset = fetch_california_housing(as_frame = True)

X = dataset.data[["MedInc", "AveRooms", "HouseAge"]]
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

ada_reg = AdaBoostRegressor(estimator = DecisionTreeRegressor(max_depth = 1),
                            n_estimators = 50,
                            learning_rate = 1.0,
                            random_state = 42)

ada_reg.fit(X_train, y_train)
y_pred = ada_reg.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R2 Score: {r2}")
print(f"MSE Score: {mse}")

'''
R2 Score: 0.33739997133259236
MSE Score: 0.8682770265484484
'''