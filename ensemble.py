import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


file_path = 'car_evaluation - car_evaluation.csv'  # Update with the correct file path
car_data = pd.read_csv(file_path)


le = LabelEncoder()
for col in car_data.columns:
    car_data[col] = le.fit_transform(car_data[col])

X = car_data.drop('outcome', axis=1)
y = car_data['outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

svc_model = SVC(kernel='rbf', probability=True, random_state=42)
svc_model.fit(X_train, y_train)
svc_pred = svc_model.predict(X_test)
svc_acc = accuracy_score(y_test, svc_pred)
print("\n🔹 SVC Accuracy:", svc_acc)
print(classification_report(y_test, svc_pred))

lr_model = LogisticRegression(max_iter=200, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
print("\n🔹 Logistic Regression Accuracy:", lr_acc)
print(classification_report(y_test, lr_pred))



rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print("\n🔹 Random Forest Accuracy:", rf_acc)
print(classification_report(y_test, rf_pred))

voting_model = VotingClassifier(estimators=[
    ('svc', svc_model),
    ('lr', lr_model),
    ('rf', rf_model)
], voting='hard')


voting_model.fit(X_train, y_train)
voting_pred = voting_model.predict(X_test)
voting_acc = accuracy_score(y_test, voting_pred)
print("\n🔹 Voting Classifier Accuracy:", voting_acc)
print(classification_report(y_test, voting_pred))

conf_matrices = {
    "SVC": confusion_matrix(y_test, svc_pred),
    "Logistic Regression": confusion_matrix(y_test, lr_pred),
    "Random Forest": confusion_matrix(y_test, rf_pred),
    "Voting Classifier": confusion_matrix(y_test, voting_pred)
}

# Accuracy Visualization
models = ['SVC', 'Logistic Regression', 'Random Forest', 'Voting Classifier']
accuracies = [svc_acc, lr_acc, rf_acc, voting_acc]

plt.figure(figsize=(12, 6))
sns.barplot(x=models, y=accuracies)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()


plt.figure(figsize=(10, 8))

for i, (model_name, matrix) in enumerate(conf_matrices.items(), 1):
    plt.subplot(2, 2, i)
    sns.heatmap(matrix, annot=True, fmt='d'
               )
    plt.title(model_name)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
plt.show()

