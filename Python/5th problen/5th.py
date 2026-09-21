import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load the dataset
data = pd.read_csv("students.csv")

# Display the dataset
print(data)


# Select input features
X = data[["Study_Hours", "Attendance"]]

# Select target
y = data["Result"]


# Convert Pass/Fail into numbers
y = y.map({"Fail": 0, "Pass": 1})


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Create the model
model = LogisticRegression()


# Train the model
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


# Predict a new student
new_student = [[6, 82]]

prediction = model.predict(new_student)


if prediction[0] == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")