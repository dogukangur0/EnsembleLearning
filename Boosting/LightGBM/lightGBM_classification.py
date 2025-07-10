from lightgbm import LGBMClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

lgbm_clf = LGBMClassifier(
    n_estimators = 200,
    learning_rate = 0.04,
    max_depth = 5,
    subsample = 0.8,
    colsample_bytree = 0.8,
    reg_alpha = 0.1,  # L1 ve overfitting'i azaltmak için
    reg_lambda = 0.2, # L2 regularization
    min_child_samples = 20, # bir yapragin bölünebilmesi için gerekli min örnek sayisi
    min_split_gain = 0.01,
    class_weight = "balanced", # sınıflar dengesizse otomatik agırlıklandırma yapar
    boosting_type = "gbdt",
    random_state = 42)

lgbm_clf.fit(X_train, y_train)

y_pred = lgbm_clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report: \n {classification_report(y_pred, y_test)}")

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot = True, fmt = "d", xticklabels = dataset.target_names, yticklabels = dataset.target_names)
plt.xlabel("Predictions")
plt.ylabel("Real")
plt.show()

'''
Accuracy: 0.9590643274853801
Classification Report: 
               precision    recall  f1-score   support

           0       0.94      0.95      0.94        62
           1       0.97      0.96      0.97       109

    accuracy                           0.96       171
   macro avg       0.95      0.96      0.96       171
weighted avg       0.96      0.96      0.96       171
'''







