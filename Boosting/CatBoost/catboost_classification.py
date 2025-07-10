from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = "diamonds.csv"
dataset = pd.read_csv(file_path)

dataset["price_category"] = (dataset["price"] > 3000).astype(int)

X = dataset.drop(columns = ["price", "price_category"])
y = dataset["price_category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

cat_clf = CatBoostClassifier(
    iterations = 500,
    learning_rate = 0.05,
    depth = 8,
    l2_leaf_reg = 5,
    loss_function = "Logloss",
    eval_metric = "Accuracy",
    cat_features = ['cut', 'color', 'clarity'],
    random_state = 42,
    verbose = 100,
    early_stopping_rounds = 30
    )

cat_clf.fit(X_train, y_train, eval_set = (X_test, y_test))
'''
0:	 learn: 0.9537758	test: 0.9521691	best: 0.9521691 (0)	total: 72.8ms	remaining: 36.3s
100: learn: 0.9820583	test: 0.9840564	best: 0.9840564 (99)	total: 6.27s	remaining: 24.8s
Stopped by overfitting detector  (30 iterations wait)

bestTest = 0.9840563589
bestIteration = 99
'''

y_pred = cat_clf.predict(X_test)

print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report: \n {classification_report(y_test, y_pred)}")

'''
Accuracy Score: 0.9840563589173156
Classification Report: 
               precision    recall  f1-score   support

           0       0.98      0.99      0.99      3065
           1       0.98      0.98      0.98      2329

    accuracy                           0.98      5394
   macro avg       0.98      0.98      0.98      5394
weighted avg       0.98      0.98      0.98      5394
'''

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot = True, fmt = "d", xticklabels = dataset["price_category"].unique(), yticklabels = dataset["price_category"].unique())
plt.xlabel("Predictions")
plt.ylabel("Real")
plt.title("CatBoost Classification")
plt.show()


















