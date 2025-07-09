from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

extra_tree_model = ExtraTreesClassifier(
                    n_estimators = 100,
                    max_depth = 10,
                    min_samples_split = 5,
                    random_state = 42)

extra_tree_model.fit(X_train, y_train)
y_pred = extra_tree_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cr = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix: \n {cm}")
print(cr)

"""
Accuracy: 0.9736842105263158
Confusion Matrix: 
 [[41  2]
 [ 1 70]]
              precision    recall  f1-score   support

           0       0.98      0.95      0.96        43
           1       0.97      0.99      0.98        71

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114
"""

plt.figure()
sns.heatmap(cm, annot = True, fmt = "d", xticklabels = dataset.target_names, yticklabels = dataset.target_names)
plt.title("Confusion Matrix")
plt.xlabel("Prediction")
plt.ylabel("Real")

# visualize feature importance

feature_importances = extra_tree_model.feature_importances_

sorted_idx = np.argsort(feature_importances)[::-1]

features = dataset.feature_names

plt.figure()
plt.bar(range(X.shape[1]), feature_importances[sorted_idx], align = "center")
plt.xticks(range(X.shape[1]), features[sorted_idx], rotation = 90)
plt.title("Feature Importances")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()


















