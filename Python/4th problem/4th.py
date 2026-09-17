import pandas as pd
import os

print(os.getcwd())

data = pd.read_csv("house_price.csv")

print(data)