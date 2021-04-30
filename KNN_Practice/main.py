import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('car.data')
#print(data.head())

X = data[[
    'buying',
    'maint',
    'safety'
]].values
y = data[['class']] #This is the label

#change the data from string to numeric value
le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = le.fit_transform(X[:, i])
#print(X)  

#change y
label_mapping = {
    'unacc' :0,
    'acc' :1,
    'good' :2,
    'vgood' :3
    }
y['class'] = y['class'].map(label_mapping)   
y = np.array(y)
#print(y)

#create model
knn= neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

knn.fit(X_train, y_train)  #train the model
prediction = knn.predict(X_test)
accuracy= metrics.accuracy_score(y_test, prediction)
print(f'Prediction : {prediction}')
print(f'Accuracy : {accuracy}')
print("type the number of instance you want to predict")
a = int(input('>'))
print("actual value:", y[a])
print("predicted value:", knn.predict(X)[a])