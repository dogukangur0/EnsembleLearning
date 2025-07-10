from sklearn.datasets import load_digits
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

dataset = load_digits()

X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

plt.figure()
fig, axes = plt.subplots(1, 2)
axes[0].imshow(dataset.images[0], cmap = "gray")
axes[0].set_title(f"Class Name: {dataset.target[0]}")
axes[0].axis("off")

axes[1].imshow(dataset.images[1], cmap = "gray")
axes[1].set_title(f"Class Name: {dataset.target[0]}")
axes[1].axis("off")


gb_clf = GradientBoostingClassifier(
            n_estimators = 150,
            learning_rate = 0.05,
            max_depth = 4,
            subsample = 0.8,
            min_samples_split = 5,
            min_samples_leaf = 3,
            max_features = "sqrt",
            validation_fraction = 0.1,
            n_iter_no_change = 5,
            random_state = 42)

gb_clf.fit(X_train, y_train)
y_pred = gb_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
class_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {acc}")
print(f"Confusion Matrix: \n {cm}")
print(f"Classification Reporst: \n {class_rep}")

'''
Accuracy: 0.9666666666666667
Confusion Matrix: 
 [[32  0  0  0  1  0  0  0  0  0]
 [ 0 28  0  0  0  0  0  0  0  0]
 [ 0  0 33  0  0  0  0  0  0  0]
 [ 0  0  0 32  0  1  0  0  1  0]
 [ 0  0  0  0 46  0  0  0  0  0]
 [ 0  0  0  0  0 44  1  0  0  2]
 [ 0  0  0  0  0  1 34  0  0  0]
 [ 0  0  0  0  0  0  0 33  0  1]
 [ 0  0  0  0  0  1  0  0 29  0]
 [ 0  0  0  0  0  1  0  1  1 37]]
 
Classification Report: 
                precision    recall  f1-score   support

            0       1.00      0.97      0.98        33
            1       1.00      1.00      1.00        28
            2       1.00      1.00      1.00        33
            3       1.00      0.94      0.97        34
            4       0.98      1.00      0.99        46
            5       0.92      0.94      0.93        47
            6       0.97      0.97      0.97        35
            7       0.97      0.97      0.97        34
            8       0.94      0.97      0.95        30
            9       0.93      0.93      0.93        40

     accuracy                           0.97       360
    macro avg       0.97      0.97      0.97       360
 weighted avg       0.97      0.97      0.97       360
'''



