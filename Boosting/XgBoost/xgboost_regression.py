from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import matplotlib.pyplot as plt

dataset = fetch_california_housing()

X = dataset.data
y = dataset.target 

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size = 0.5, random_state = 42)

xgb_reg = XGBRegressor(
           n_estimators = 200,
           learnin_rate = 0.05,
           max_depth = 6,
           subsample = 0.8,
           colsample_bytree = 0.8,
           eval_metrics = "rmse",
           random_state = 42,
           early_stopping_rounds = 10)


xgb_reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose = True)
y_pred = xgb_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE Score: {np.sqrt(mse)}")
print(f"R2 Score: {r2}")


'''
[135] validation_0-rmse:0.47332
RMSE Score: 0.44727105068598644
R2 Score: 0.8480736590202752
'''

plt.figure()
plt.scatter(y_test, y_pred, alpha = 0.3)
plt.plot([0,5], [0,5], color = "red")
plt.xlabel("Real")
plt.ylabel("Prediction")
plt.title("XGBoost Regressor")
plt.grid(True)
plt.show()













