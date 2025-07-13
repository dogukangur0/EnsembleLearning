from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np

dataset = fetch_california_housing()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

base_models = [
    ('dtr', DecisionTreeRegressor(max_depth = 5, random_state = 42)),
    ('svr', SVR(kernel = "rbf", C = 100))
    ]

meta_model = Ridge(alpha = 1)

starcking_reg = StackingRegressor(estimators = base_models,
                                  final_estimator = meta_model,
                                  cv = 5)

starcking_reg.fit(X_train, y_train)

y_pred = starcking_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2_score = r2_score(y_test, y_pred)

print(f"RMSE Score: {rmse}")
print(f"R2_Score: {r2_score}")

'''
RMSE Score: 0.7100290316080753
R2_Score: 0.6219723391925293
'''