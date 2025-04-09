import matplotlib as plt
import pandas as pd
df=pd.read_csv('breastcancer.csv')
df.head()
df.keys()
df.info()


df.drop(columns=['id'],inplace=True)
df.keys()
df.head()


df.shape


from sklearn.model_selection import train_test_split
x=df.iloc[:,1:]
y=df.iloc[:,0:1]
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=2)


X_train.shape


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)


from sklearn.metrics import accuracy_score
y_pred = knn.predict(X_test)
accuracy_score(y_test, y_pred)


from sklearn.metrics import confusion_matrix
import seaborn as sns
cm= confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)