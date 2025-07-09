from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = fetch_california_housing()

X = dataset.data 
y = dataset.target 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

models = {"Bagging Regressor": BaggingRegressor(estimator=DecisionTreeRegressor(),
                                                n_estimators = 100,
                                                max_samples = 0.8,
                                                max_features = 0.8,
                                                random_state = 42),
          "RandomForest Regressor": RandomForestRegressor(n_estimators=100,
                                                          max_depth=15,
                                                          min_samples_split=5,
                                                          random_state = 42),
          "ExtraTrees Regressor": ExtraTreesRegressor(n_estimators = 100,
                                                      max_depth=15,
                                                      min_samples_split=5,
                                                      random_state=42)}

results = {}
predictions = {}
for name, model in tqdm(models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MSE":mse, "R2":r2}
    predictions[name] = y_pred
    
results_df = pd.DataFrame(results).T

plt.figure()
for i, (name, y_pred) in enumerate(predictions.items()):
    plt.subplot(1, 3, i+1)
    plt.scatter(y_test, y_pred, alpha = 0.5, label = name)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw = 2)
    plt.title(f"{name} Gercek vs Tahmin")
    plt.xlabel("Gercek")
    plt.ylabel("Tahmin")
    plt.legend()
plt.tight_layout()
plt.show()    


plt.figure()
for i, (name, y_pred) in enumerate(predictions.items()):
    residuals = y_test - y_pred
    plt.subplot(1, 3, i+1)
    plt.scatter(y_pred, residuals, alpha = 0.5, label = name)
    plt.axhline(y = 0, color = "r", linestyle = "--")
    plt.title(f"{name} Gercek vs Tahmin")
    plt.xlabel("Gercek")
    plt.ylabel("Tahmin")
    plt.legend()
plt.tight_layout()
plt.show()    


feature_names = dataset.feature_names
for i, (name, model) in enumerate(models.items()):
    if hasattr(model, "feature_importances_"):
        feature_importance = model.feature_importances_
        sorted_idx = np.argsort(feature_importance)[::-1]
        plt.subplot(1, 3, i+1)
        plt.bar(range(X.shape[1]), feature_importance[sorted_idx], label = name)
        plt.xticks(range(X.shape[1]), np.array(feature_names)[sorted_idx], rotation = 45)
        plt.title(f"{name} Feature Importance")
        plt.xlabel("Features")
        plt.ylabel("Importance Score")
        plt.legend()
plt.tight_layout()
plt.show()





















