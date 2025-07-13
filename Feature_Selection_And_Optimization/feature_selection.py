from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
ada_clf = AdaBoostClassifier(estimator = DecisionTreeClassifier(max_depth = 1),
                         n_estimators = 100,
                         learning_rate = 0.5,
                         random_state = 42)

ada_clf.fit(X_train, y_train)
y_pred = ada_clf.predict(X_test)
print(f"AdaBoost Test Acc: {accuracy_score(y_test, y_pred)}")
# AdaBoost Test Acc: 0.9649122807017544

feature_importance = ada_clf.feature_importances_
selector = SelectFromModel(ada_clf, threshold = "mean", prefit = True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

ada_clf_selected = AdaBoostClassifier(estimator = DecisionTreeClassifier(max_depth = 1),
                                  n_estimators = 100,
                                  learning_rate = 0.5,
                                  random_state = 42)

ada_clf_selected.fit(X_train_selected, y_train)
y_pred_selected = ada_clf_selected.predict(X_test_selected)
print(f"AdaBoost Feature Selection Test Acc: {accuracy_score(y_test, y_pred_selected)}")
# AdaBoost Feature Selection Test Acc: 0.9649122807017544