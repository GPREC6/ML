import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree

import pandas as pd
file_path = "iris - iris.csv"
df = pd.read_csv(file_path)


print(df.head())

X = df.iloc[:, :-1] 
y = df.iloc[:, -1] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

plt.figure(figsize=(12, 8))
tree.plot_tree(model, 
               feature_names=X.columns, 
               class_names=y.astype(str).unique(), 
               filled=True)
plt.show()