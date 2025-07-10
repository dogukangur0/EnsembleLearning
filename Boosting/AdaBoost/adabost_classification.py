from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

dataset = load_iris()

X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

estimators = [1, 5, 10, 20, 50, 100]

acc_scores = []
cm_scores = []
for estimator in estimators:
    ada_clf = AdaBoostClassifier(n_estimators = estimator, learning_rate = 1, random_state = 42)
    ada_clf.fit(X_train, y_train)
    
    y_pred = ada_clf.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    cm_scores.append(cm)
    acc_scores.append(accuracy)

plt.figure()
plt.plot(estimators, acc_scores, marker = "o")
plt.xticks(estimators)
plt.show()

i = 0
plt.figure()
for estimator, cm_score, acc_score in zip(estimators,cm_scores, acc_scores):
    plt.subplot(2, 3, i+1)
    sns.heatmap(cm_score, annot = True, fmt = "d", cmap = "Blues", xticklabels = dataset.target_names, yticklabels = dataset.target_names)
    plt.title(f"Estimator Number: {estimator}, Accuracy Score: {acc_score:.3f}")
    plt.xlabel("Predictions")
    plt.ylabel("Real")
    i = i+1
plt.subplots_adjust(hspace=0.5) 
plt.legend()
    
    
'''
estimators : [1, 5, 10, 20, 50, 100]
acc_scores : [0.63, 1.0, 1.0, 1.0, 0.93, 0.93]
cm_scores  :
           [array([[10,  0,  0],
                   [ 0,  9,  0],
                   [ 0, 11,  0]]),
            array([[10,  0,  0],
                   [ 0,  9,  0],
                   [ 0,  0, 11]]),
            array([[10,  0,  0],
                   [ 0,  9,  0],
                   [ 0,  0, 11]]),
            array([[10,  0,  0],
                   [ 0,  9,  0],
                   [ 0,  0, 11]]),
            array([[10,  0,  0],
                   [ 0,  8,  1],
                   [ 0,  1, 10]]),
            array([[10,  0,  0],
                   [ 0,  8,  1],
                   [ 0,  1, 10]])]
'''
    
    
    
    
    
    
    
    
