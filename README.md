# Importing the necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


In this code, we first import the necessary libraries: load_iris from sklearn.datasets to load the Iris dataset, train_test_split from sklearn.model_selection to split the dataset into training and testing sets, KNeighborsClassifier from sklearn.neighbors to create a K-nearest neighbors classifier, and accuracy_score from sklearn.metrics to calculate the accuracy of the model.

We then load the Iris dataset using load_iris() and split it into training and testing sets using train_test_split(). Next, we create a K-nearest neighbors classifier with KNeighborsClassifier(n_neighbors=3) and train it on the training set using fit(). We then make predictions on the test set using predict() and calculate the accuracy of the model using accuracy_score().

Finally, we print the accuracy of the model on the test set.

This code provides a basic example of training a machine learning model using scikit-learn. You can modify it to experiment with different algorithms, datasets, and parameters according to your needs.

Remember to install scikit-learn (pip install scikit-learn) before running this code.
