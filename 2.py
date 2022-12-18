import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\Mark0\OneDrive\Рабочий стол\Boston.csv')

X = df["lstat"].values
y = df["medv"].values
X = X.reshape(-1, 1)

reg = LinearRegression()
reg.fit(X, y)
predictions = reg.predict(X)

plt.scatter(X, y, color="blue")
plt.plot(X, predictions, color="red")
plt.xlabel("lower status of the population (%)")
plt.ylabel("median value of owner-occupied homes in ($)")
plt.show()
