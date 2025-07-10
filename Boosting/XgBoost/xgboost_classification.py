from xgboost import XGBClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt

dataset = load_digits()
X = dataset.data
y = dataset.target

fig, axes = plt.subplots(1, 2, figsize = (8,4))
for i, ax in enumerate(axes):
    ax.imshow(dataset.images[i], cmap = "gray")
    ax.set_title(f"Class Name: {dataset.target[i]}")
    ax.axis("off")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

xgb_clf = XGBClassifier(
    n_estimators = 150,
    learning_rate = 0.05,
    max_depth = 4,
    subsample = 0.8,
    colsample_bytree = 0.5,
    min_child_weight = 3,
    gamma = 0,
    early_stopping_rounds = 5,
    eval_metrics = "mlogloss",
    random_state = 42,
    use_label_encoder = False)

xgb_clf.fit(X_train, y_train, eval_set = [(X_test, y_test)], verbose = True)
y_pred = xgb_clf.predict(X_test)
acc_score = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Acc Score: {acc_score}")
print(f"Confusion Matrix: \n {cm}")

'''
Acc Score: 0.9666666666666667
Confusion Matrix: 
 [[32  0  0  0  1  0  0  0  0  0]
 [ 0 28  0  0  0  0  0  0  0  0]
 [ 0  0 33  0  0  0  0  0  0  0]
 [ 0  0  0 33  0  1  0  0  0  0]
 [ 0  0  0  0 46  0  0  0  0  0]
 [ 0  0  0  0  0 44  1  0  0  2]
 [ 0  0  0  0  0  1 34  0  0  0]
 [ 0  0  0  0  0  0  0 33  0  1]
 [ 0  1  0  0  0  1  0  0 28  0]
 [ 0  0  0  0  0  0  0  1  2 37]]
'''






















