import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Training data
data = {
    "hours": [1, 2, 3, 4, 5, 6, 7],
    "result": ["Fail", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass"]
}

# Convert data into DataFrame
df = pd.DataFrame(data)

# Input and output
X = df[["hours"]]
y = df["result"]

# Create the ML model
model = DecisionTreeClassifier()

# Train the model
model.fit(X, y)

# Make prediction
prediction = model.predict([[5]])

# Display prediction
print("Prediction:", prediction[0])