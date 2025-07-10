from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

dataset = load_diabetes()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

gb_reg = GradientBoostingRegressor(
    n_estimators = 200,
    learning_rate = 0.01,
    max_depth = 5,
    subsample = 0.8,
    min_samples_split = 5,
    min_samples_leaf = 4,
    validation_fraction = 0.1,
    n_iter_no_change = 5,
    random_state = 42)

gb_reg.fit(X_train, y_train)
y_pred = gb_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE Score: {mse}")
print(f"R2 Score: {r2}")

residuals = y_test - y_pred
plt.figure()
plt.scatter(y_pred, residuals, alpha = 0.3)
plt.axhline(0, color = "r")
plt.title("GB - Hata Dagilimi")
plt.xlabel("Predictions")
plt.ylabel("Hata")
plt.legend()


'''
MSE Score: 3051.6791497053932
R2 Score: 0.42401061831424225
'''