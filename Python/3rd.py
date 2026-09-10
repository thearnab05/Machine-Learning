import pandas as pd 
from sklearn.linear_model import LinearRegression

data = {
    "size": [500,750,1000,1250,1500,1750,2000],
    "price": [20,30,40,50,60,70,80]
}

df = pd.DataFrame(data)

x = df["size"]
y = df["price"]

model = LinearRegression()

model.fit(x,y)

prediction = model.predict([[1600]])

print("Predicted price:", prediction[0], "lakh")