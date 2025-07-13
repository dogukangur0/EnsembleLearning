from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

base_models = [
    ('dt', DecisionTreeClassifier(max_depth = 5, random_state = 42)),
     ('svc', SVC(probability = True, C = 1, kernel = "rbf", random_state = 42))
    ]

meta_model = LogisticRegression()

stacking_clf = StackingClassifier(estimators = base_models,
                                  final_estimator = meta_model,
                                  cv = 5,
                                  stack_method = "predict_proba")


stacking_clf.fit(X_train, y_train)

y_pred = stacking_clf.predict(X_test)

print(f"Stacking Classifier Acc: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report: \n {classification_report(y_test, y_pred)}")

'''
Stacking Classifier Acc: 0.9649122807017544
Classification Report: 
               precision    recall  f1-score   support

           0       0.98      0.93      0.95        43
           1       0.96      0.99      0.97        71

    accuracy                           0.96       114
   macro avg       0.97      0.96      0.96       114
weighted avg       0.97      0.96      0.96       114
'''
