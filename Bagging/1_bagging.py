# import libraries
from sklearn.ensemble import BaggingClassifier # bagging model
from sklearn.tree import DecisionTreeClassifier # weak learner to be used in bagging 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
 
# download dataset -> iris dataset
dataset = load_iris()
X = dataset.data
y = dataset.target

# data train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# define base model : decision tree
base_model = DecisionTreeClassifier(random_state = 42)

# create bagging model
bagging_model = BaggingClassifier(estimator = base_model,
                                  n_estimators = 10,
                                  max_samples = 0.8,
                                  max_features = 0.8,
                                  bootstrap = True, # allows selection same samples
                                  random_state = 42)
# model training
bagging_model.fit(X_train, y_train)
# model testing
y_pred = bagging_model.predict(X_test)
# evaluate model accuray
accuracy = accuracy_score(y_test, y_pred)
print(accuracy) # 1.0
# make example