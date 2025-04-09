import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


data = pd.read_csv('spamham.csv',encoding='latin-1')
data.head()


data.info()

data.shape


X = data.email
y = data.type
vect = TfidfVectorizer()
x = vect.fit_transform(X) 
feature_name = vect.get_feature_names()
feature_name[2000:2050]

len(feature_name)

x

x = x.toarray()
x

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


model = BernoulliNB()
model.fit(x_train,y_train)
y_predict = model.predict(x_test)
accuracy_score(y_test,y_predict)


model.score(x_train,y_train)

new_data = pd.Series('XXX MobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap. www.fraud.com?n=QJKGIGHJJGCBLMessage us to Bz-10567')
new_data


new_data = vect.transform(new_data)
new_data


model.predict(new_data)


array(['spam'], dtype='<U4')