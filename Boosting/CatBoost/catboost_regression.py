from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

file_name = "diamonds.csv"
dataset = pd.read_csv(file_name)

X = dataset.drop(columns = ["price"])
y = dataset["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

cat_reg = CatBoostRegressor(
    iterations = 500,
    learning_rate = 0.05,
    depth = 8,
    l2_leaf_reg = 4,
    loss_function = "RMSE",
    cat_features = ["cut", "color", "clarity"],
    random_state = 42,
    verbose = 100,
    early_stopping_rounds = 50
    )

cat_reg.fit(X_train, y_train,eval_set = (X_test, y_test))

'''
0:	learn: 3816.4255258	test: 3844.6707781	best: 3844.6707781 (0)	total: 49.7ms	remaining: 24.8s
100:	learn: 594.6088367	test: 597.7989732	best: 597.7989732 (100)	total: 4.41s	remaining: 17.4s
200:	learn: 539.1517510	test: 552.6639366	best: 552.6639366 (200)	total: 8.94s	remaining: 13.3s
300:	learn: 522.5367142	test: 544.2039429	best: 544.2039429 (300)	total: 13.3s	remaining: 8.82s
400:	learn: 508.0933969	test: 539.7647391	best: 539.7647391 (400)	total: 17.7s	remaining: 4.37s
499:	learn: 494.3727648	test: 535.0495719	best: 535.0300425 (498)	total: 22.3s	remaining: 0us

bestTest = 535.0300425
bestIteration = 498
'''

y_pred = cat_reg.predict(X_test)

print(f"RMSE Score: {np.sqrt(mean_squared_error(y_test, y_pred))}")
print(f"R2 Score: {r2_score(y_test, y_pred)}")
      
'''
RMSE Score: 535.0300424908952
R2 Score: 0.9822712378674515
'''

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(y_test, y_pred, alpha = 0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color = "red")
plt.xlabel("Real")
plt.ylabel("Prediction")
plt.title("CatBoost Regression")
plt.show()      
      
      
      
      
      
      
      
      
      
      
      
      
      