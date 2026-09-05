import pandas as pd 
from sklearn.tree import DecisionTreeClassfier

data = {
    "hours" : [1, 2, 3, 4, 5, 6, 7, 8],
    "attendance" : [50,55,60,65,70,75,80,90],
    "result" : ["Fail","Fail","Fail","Pass","Pass","Pass","Pass","Pass"]
}

df = pd.DataFrame(data)

X = df[["hours", "attendance"]]
y = df["result"]

model = DecisionTreeClassifier()

model.fit(X, y)

prediction = model.predict([[5, 72]])
