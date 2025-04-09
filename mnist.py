from sklearn import datasets
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt

mnist_datasets = datasets.load_digits()

print(mnist_datasets.keys())

X_data, Y_data = mnist_datasets['data'], mnist_datasets['target']

X_data.shape
Y_data.shape

first_digit = X_data[1]
first_digit_image = first_digit.reshape(8, 8)

plt.imshow(first_digit_image)
plt.axis('off')  
plt.show()

X_train_data, X_test_data, Y_train_data, Y_test_data = (
    X_data[:1600], X_data[1600:], Y_data[:1600], Y_data[1600:]
)

X_train_data.shape
X_test_data.shape
Y_train_data.shape
Y_test_data.shape

Y_predict_digit = (Y_train_data == 1)

sgd_classifier = SGDClassifier(random_state=42)
sgd_classifier.fit(X_train_data, Y_predict_digit)

sgd_classifier.predict([first_digit])