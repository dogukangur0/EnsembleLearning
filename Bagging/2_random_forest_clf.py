# import libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
# load dataset: breast cancer
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target
# split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# create random-forest model
rf_model = RandomForestClassifier(n_estimators=100, # ağaç sayısı
                                    max_depth = 10, # maximum derinlik
                                    min_samples_split = 5, # bir düğümü bölmek için minimum örnek sayısı
                                    random_state = 42)
# model training
rf_model.fit(X_train, y_train)
# model testing
y_pred = rf_model.predict(X_test)
# evaluation : accuracy
accuracy = accuracy_score(y_test, y_pred)
classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

'''
Accuracy: 0.9649122807017544
              precision    recall  f1-score   support

           0       0.98      0.93      0.95        43
           1       0.96      0.99      0.97        71

    accuracy                           0.96       114
   macro avg       0.97      0.96      0.96       114
weighted avg       0.97      0.96      0.96       114
'''