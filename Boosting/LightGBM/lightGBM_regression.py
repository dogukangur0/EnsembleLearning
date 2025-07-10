from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

file_path = "energy_efficiency.csv"

dataset = pd.read_csv(file_path)

X = dataset.drop(columns = ['heating_load', 'cooling_load'])
y = dataset["heating_load"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

lgbm_reg = LGBMRegressor(
    n_estimators = 200, 
    learning_rate = 0.05,
    max_depth = 5,
    num_leaves = 30, # yaprak sayısı
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_samples = 20,
    reg_alpha = 0.1,
    reg_lambda = 0.2,
    random_state = 42)

lgbm_reg.fit(X_train, y_train, eval_set = [(X_test, y_test)])

y_pred = lgbm_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE Score: {np.sqrt(mse)}")
print(f"R2 Score: {r2}")

'''
RMSE Score: 0.4763827028522075
R2 Score: 0.9978227455247326
'''

plt.figure()
plt.scatter(y_test, y_pred, alpha = 0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color = "red")
plt.xlabel("Real")
plt.ylabel("Prediction")
plt.title("Energy Efficiency")
plt.show()








