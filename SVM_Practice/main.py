from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()

#split the data into features and labels
X = iris.data
y = iris.target

classes = ['Tris Setosa', 'Iris Versicolour', 'Iris Virginica']
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#create the model
model = svm.SVC()
model.fit(X_train, y_train)

print(model)
prediction = model.predict(X_test)
accura = accuracy_score(y_test, prediction)

print("prediction:", prediction)
print("actual value:", y_test)
print("accuracy:", accura)

#print class name
for i in range(len(prediction)):
    print(classes[prediction[i]])