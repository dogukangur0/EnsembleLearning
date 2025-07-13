from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

import numpy as np

dataset = fetch_california_housing()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model1 = LinearRegression()
model2 = DecisionTreeRegressor(max_depth = 8)
model3 = SVR(kernel = "rbf", C = 1)

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred3 = model3.predict(X_test)

y_pred_avg = (y_pred1 + y_pred2 + y_pred3) / 3
rmse = np.sqrt(mean_squared_error(y_test, y_pred_avg))

print(f"RMSE: {rmse}")