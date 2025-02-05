# train_model.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the Iris dataset
iris = load_iris()
X = X = [
    [5.1, 3.5, 1.4, 0.2],  # Sample 1
    [4.9, 3.0, 1.4, 0.2],  # Sample 2
    [5.9, 3.0, 5.1, 1.8],  # Sample 150
]
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'iris_model.pkl')