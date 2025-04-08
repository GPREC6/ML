
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


data = pd.read_csv('iris - iris.csv')
data.head()

from collections import Counter
Counter(data.target)

X = data.iloc[:,:-1]
y = data[['target']]

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=10,test_size=0.2)

model = SVC()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
accuracy_score(y_test,y_predict)

print(classification_report(y_test,y_predict))

confusion_matrix(y_test,y_predict)

model = SVC(C=0.1)
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
accuracy_score(y_test,y_predict)



model = SVC(C=10)
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
accuracy_score(y_test,y_predict)

model = SVC(C=100)
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
accuracy_score(y_test,y_predict)

model = SVC(C=1000)
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
accuracy_score(y_test,y_predict)

#gamma
model = SVC(gamma=0.01)
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
accuracy_score(y_test,y_predict)

model = SVC(gamma=0.1)
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
accuracy_score(y_test,y_predict)

model = SVC(gamma=1)
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
accuracy_score(y_test,y_predict)

model = SVC(gamma=10)
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
accuracy_score(y_test,y_predict)

model = SVC(gamma=100)
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
accuracy_score(y_test,y_predict)

model = SVC(C=100,gamma=0.1,kernel='linear')
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
accuracy_score(y_test,y_predict)

#Kernel
model = SVC(kernel='linear')
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
accuracy_score(y_test,y_predict)

model = SVC(kernel='poly')
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
accuracy_score(y_test,y_predict)


df0 = data[:50]
df1 = data[50:100]
df2 = data[100:150]

x = data.iloc[:100,:2]
y = data.target[:100]

plt.scatter(df0['sepal_length'],df0['sepal_width'],color='green',marker='+')
plt.scatter(df2['sepal_length'],df2['sepal_width'],color='red',marker='*')
plt.scatter(df1['sepal_length'],df1['sepal_width'],color='blue',marker='*')

plt.scatter(df0['petal_length'],df0['petal_width'],color='green',marker='+')
plt.scatter(df2['petal_length'],df2['petal_width'],color='red',marker='*')
plt.scatter(df1['petal_length'],df1['petal_width'],color='blue',marker='*')

import seaborn as sns
sns.pairplot(data,hue="target", palette="CMRmap")


