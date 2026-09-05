import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = {
    "hours": [1, 2, 3, 4, 5, 6, 7],
    "result": ["Fail", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass"]
}

df = pd.DataFrame(data)

print(df)

x = df[["hours"]]
y = df["result"]