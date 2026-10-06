import pickle

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target


# Create model
model = LogisticRegression(max_iter=200)


# Train model
model.fit(X, y)


# Save model
with open("iris.pkl", "wb") as file:
    pickle.dump(model, file)


print("Model saved successfully!")