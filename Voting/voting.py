from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dataset = load_iris()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model1 = LogisticRegression(max_iter = 200)
model2 = DecisionTreeClassifier(max_depth = 5)
model3 = SVC(probability = True)

voting_clf_hard = VotingClassifier(
    estimators = [('lr', model1), ('dtc', model2), ('svc', model3)],
    voting = "hard")

# voting : hard -> çoğunluk, soft -> olasılıksal

voting_clf_hard.fit(X_train, y_train)
y_pred = voting_clf_hard.predict(X_test)

print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")

# Accuracy Score: 1.0

voting_clf_soft = VotingClassifier(
    estimators = [('lr', model1), ('dtc', model2), ('svc', model3)],
    voting = "soft")

# voting : hard -> çoğunluk, soft -> olasılıksal

voting_clf_soft.fit(X_train, y_train)
y_pred = voting_clf_soft.predict(X_test)

print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")

# Accuracy Score: 1.0