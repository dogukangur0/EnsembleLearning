from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

base_model = DecisionTreeClassifier(max_depth = 1)

ada_clf = AdaBoostClassifier(estimator = base_model, random_state = 42)

param_grid = {"n_estimators": [50, 100, 200],
              "learning_rate": [0.01, 0.05, 0.01, 0.1, 1],
              "estimator__max_depth": [1, 2, 3]}

grid_search = GridSearchCV(ada_clf, param_grid, cv = 5, verbose = 1, scoring = "accuracy", n_jobs = -1)

grid_search.fit(X_train, y_train)

print(f"En iyi parametre: {grid_search.best_estimator_}")

# En iyi parametre: AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), learning_rate=1, random_state=42)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(f"AdaBoost Grid Search Accuracy: {accuracy_score(y_test, y_pred)}")
# AdaBoost Grid Search Accuracy: 0.9649122807017544